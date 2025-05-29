import time

import torch
import torch.nn as nn

from sparsegpt import *
from model_utils import *
# from quant import *
from qwen_utils import qwen_fuse_rms_single_layer, RotatorOptimizer_wanda, RotatorOptimizer_sparsegpt, RotatorOptimizer_magnitude, qwen_fuse_rotation_single_layer, replace_qwen_layer

from torch.cuda.amp import GradScaler, autocast

from lm_eval_utils import evaluate_zero_shot

from gpu_utils import distribute_model

try:
    import wandb
    has_wandb = True
except:
    has_wandb = False

import os

def get_qwen(model):
    import torch
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(model, torch_dtype='auto')
    model.seqlen = 2048
    return model

@torch.no_grad()
def qwen_sequential(model, dataloader, dev):
    print("Starting...")

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.rotary_emb = model.model.rotary_emb.to(dev)
    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0, "attention_mask": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            cache["position_embeddings"] = kwargs["position_embeddings"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]
    position_embeddings = cache["position_embeddings"]

    print("Ready.")

    quantizers = {}
    for i in range(len(layers)):

        layer = layers[i].to(dev)

        for param in layer.parameters():
            param.requires_grad_(False)

        qwen_fuse_rms_single_layer(layer)

        full = find_layers(layer)

        names = list(full.keys())
        subset = {n: full[n] for n in names}

        gpts = {}
        for name in subset:
            if (
                not (args.minlayer <= i < args.maxlayer and args.prune_only in name)
            ) == (not args.invert):
                continue
            gpts[name] = SparseGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)

            return tmp

        handles = []
        for name in subset:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_embeddings=position_embeddings)[0]
        for h in handles:
            h.remove()

        weight_dict_list = [find_layers(layer)]
        hessian_dict = {}
        for name in subset:
            hessian_dict[name] = gpts[name].H
        hessian_dict_list = [hessian_dict]

        with torch.enable_grad():
            scaler = GradScaler()

            if args.method == "wanda":
                R_model = RotatorOptimizer_wanda(weight_dict_list, model.config.hidden_size, num_key_value_heads=model.config.num_key_value_heads, head_dim=model.config.hidden_size//model.config.num_attention_heads, device=dev, positive=True, hessian_dict_list=hessian_dict_list)
            elif args.method == "sparsegpt":
                R_model = RotatorOptimizer_sparsegpt(weight_dict_list, model.config.hidden_size, num_key_value_heads=model.config.num_key_value_heads, head_dim=model.config.hidden_size//model.config.num_attention_heads, device=dev, positive=True, hessian_dict_list=hessian_dict_list, percdamp=args.percdamp)
            elif args.method == "magnitude":
                R_model = RotatorOptimizer_magnitude(weight_dict_list, model.config.hidden_size, num_key_value_heads=model.config.num_key_value_heads, head_dim=model.config.hidden_size//model.config.num_attention_heads, device=dev, positive=True, hessian_dict_list=hessian_dict_list)

            R_model = R_model.to(dev)

            opt = torch.optim.Adam(R_model.parameters(), lr=0.01)

            min_loss = R_model()
            min_R1_list = R_model.get_R1_list()
            min_R2_list_list = R_model.get_R2_list_list()

            for epoch in range(args.rotation_epoch):
                if args.num_batch != 1:
                    epoch_start_time = time.time()
                    indices_hidden = torch.randperm(model.config.hidden_size)
                    indices_intermediate = torch.randperm(model.config.intermediate_size)
                    for idx_batch in range(args.num_batch):
                        indices_dict = {"hidden": indices_hidden[idx_batch * (model.config.hidden_size // args.num_batch) : (idx_batch+1) * (model.config.hidden_size // args.num_batch)], "intermediate": indices_intermediate[idx_batch * (model.config.intermediate_size // args.num_batch) : (idx_batch+1) * (model.config.intermediate_size // args.num_batch)]}
                        opt.zero_grad()
                        loss = R_model(indices_dict)
                        scaler.scale(loss).backward()
                        scaler.unscale_(opt)
                        torch.nn.utils.clip_grad_norm_(R_model.parameters(), 1.0)
                        scaler.step(opt)
                        scaler.update()
                    epoch_end_time = time.time()
                    if epoch % 5 == 0:
                        print(f"Time: {epoch_end_time - epoch_start_time:.4f} 秒")
                        loss = R_model()
                        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
                        scaler.scale(loss).backward()
                        scaler.unscale_(opt)
                        torch.nn.utils.clip_grad_norm_(R_model.parameters(), 1.0)
                        scaler.step(opt)
                        scaler.update()
                else:
                    epoch_start_time = time.time()
                    opt.zero_grad()
                    loss = R_model()
                    if epoch == 0:
                        print(f"Initial Loss: {loss}")

                    if loss < min_loss:
                        min_loss = loss
                        min_R1_list = R_model.get_R1_list()
                        min_R2_list_list = R_model.get_R2_list_list()
                        torch.cuda.empty_cache()
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(R_model.parameters(), 1.0)
                    scaler.step(opt)
                    scaler.update()
                    epoch_end_time = time.time()
                    if epoch % 10 == 0:
                        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
                        print(f"Time: {epoch_end_time - epoch_start_time:.4f} 秒")

        del R_model
        del weight_dict_list
        torch.cuda.empty_cache()

        R1 = min_R1_list[0].to(dtype).T

        R2_list = min_R2_list_list[0]
        for k in range(len(R2_list)):
            R2_list[k] = R2_list[k].to(dtype)
        qwen_fuse_rotation_single_layer(layer, R1, R2_list)

        del min_R1_list
        del min_R2_list_list
        torch.cuda.empty_cache()

        layers[i] = layer
        replace_qwen_layer(model, i, R1)
        layer = layers[i].to(dev)
        del gpts
        torch.cuda.empty_cache()

        full = find_layers(layer)

        if args.true_sequential:
            sequential = [
                ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
                ["self_attn.o_proj"],
                ["mlp.up_proj", "mlp.gate_proj"],
                ["mlp.down_proj"],
            ]
        else:
            sequential = [list(full.keys())]

        for names in sequential:
            subset = {n: full[n] for n in names}
            gpts = {}
            for name in subset:
                if (
                    not (args.minlayer <= i < args.maxlayer and args.prune_only in name)
                ) == (not args.invert):
                    continue
                gpts[name] = SparseGPT(subset[name])
                # if args.wbits < 16:
                #     gpts[name].quantizer = Quantizer()
                #     gpts[name].quantizer.configure(
                #         args.wbits, perchannel=True, sym=False, mse=False
                #     )

            def add_batch(name):
                def tmp(_, inp, out):
                    gpts[name].add_batch(inp[0].data, out.data)

                return tmp

            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            for j in range(args.nsamples):
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_embeddings=position_embeddings)[0]
            for h in handles:
                h.remove()

            for name in subset:
                print(i, name)
                print("Pruning ...  method: ", args.method)
                sparsity = args.sparsity
                if args.method == "wanda":
                    gpts[name].wanda_prune(
                        sparsity,
                        prune_n=args.prunen,
                        prune_m=args.prunem,
                    )
                elif args.method == "sparsegpt":
                    gpts[name].sparsegpt_prune(
                        sparsity,
                        prune_n=args.prunen,
                        prune_m=args.prunem,
                        percdamp=args.percdamp,
                        blocksize=args.blocksize,
                    )
                elif args.method == "magnitude":
                    gpts[name].magnitude_prune(
                        sparsity,
                        prune_n=args.prunen,
                        prune_m=args.prunem,
                    )
                gpts[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_embeddings=position_embeddings)[0]

        layers[i] = layer.cpu()
        del layer
        del gpts
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache

    return quantizers


@torch.no_grad()
def qwen_eval_ppl(model, testenc, dev,  dataset: str, log_wandb: bool = False):
    print("Evaluating ...")

    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)

    model.model.rotary_emb = model.model.rotary_emb.to(dev)

    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0, "attention_mask": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            cache["position_embeddings"] = kwargs["position_embeddings"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]
    position_embeddings = cache["position_embeddings"]

    for i in range(len(layers)):
        print(i)

        layer = layers[i].to(dev)

        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_embeddings=position_embeddings)[0]

        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if model.model.norm is not None:
        model.model.norm = model.model.norm.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if model.model.norm is not None:
            hidden_states = model.model.norm(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(f"Perplexity: {ppl.item():3f}")
    if log_wandb:
        wandb.log({f"{dataset}/perplexity": ppl.item()})

    model.config.use_cache = use_cache


if __name__ == "__main__":

    import argparse
    from data_utils import *

    parser = argparse.ArgumentParser()

    parser.add_argument("model", type=str, help="Qwen model to load")
    parser.add_argument(
        "dataset",
        type=str,
        choices=["wikitext2", "ptb", "c4"],
        help="Where to extract calibration data from.",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed for sampling the calibration data."
    )
    parser.add_argument(
        "--nsamples", type=int, default=128, help="Number of calibration data samples."
    )
    parser.add_argument(
        "--percdamp",
        type=float,
        default=0.01,
        help="Percent of the average Hessian diagonal to use for dampening.",
    )
    parser.add_argument("--sparsity", type=float, default=0, help="Target sparsity")
    parser.add_argument("--prunen", type=int, default=0, help="N for N:M pruning.")
    parser.add_argument("--prunem", type=int, default=0, help="M for N:M pruning.")
    parser.add_argument(
        "--blocksize",
        type=int,
        default=128,
        help="Blocksize to use for adaptive mask selection.",
    )
    parser.add_argument(
        "--gmp", action="store_true", help="Whether to run the GMP baseline."
    )
    parser.add_argument(
        "--wbits", type=int, default=16, help="Whether to quantize as well."
    )
    parser.add_argument(
        "--minlayer", type=int, default=-1, help="Prune all layers with id >= this."
    )
    parser.add_argument(
        "--maxlayer", type=int, default=1000, help="Prune all layers with id < this."
    )
    parser.add_argument(
        "--prune_only",
        type=str,
        default="",
        help="Prune only layers that contain this text.",
    )
    parser.add_argument("--invert", action="store_true", help="Invert subset.")
    parser.add_argument("--save", type=str, default="", help="Path to saved model.")
    parser.add_argument(
        "--true-sequential",
        action="store_true",
        help="Whether to run in true sequential model.",
    )
    parser.add_argument(
        "--log_wandb", action="store_true", help="Whether to log to wandb."
    )
    parser.add_argument(
        "--rotation_dir", type=str, default="", help="Path to saved rotation matrix."
    )
    parser.add_argument(
        "--rotation_epoch", type=int, default=100
    )
    parser.add_argument(
        "--method", type=str, choices=["magnitude", "sparsegpt", "wanda"], default="wanda"
    )
    parser.add_argument(
        "--num_batch", type=int, default=1
    )
    parser.add_argument("--distribute_model", type=str, default="False")

    args = parser.parse_args()

    # init W&B logging
    if args.log_wandb:
        assert has_wandb, "wandb not installed try `pip install wandb`"
        wandb.init(config=args)

    if not os.path.exists(args.rotation_dir):
        os.makedirs(args.rotation_dir)

    model = get_qwen(args.model)
    model.eval()

    for param in model.parameters():
        param.requires_grad_(False)

    dataloader, testloader = get_loaders(
        args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=model.seqlen
    )

    DEV = torch.device("cuda:0")
    if (args.sparsity or args.prunen) and not args.gmp:
        tick = time.time()
        qwen_sequential(model, dataloader, DEV)
        for n, p in model.named_parameters():
            print(n, torch.mean((p == 0).float()))
            if 'down_proj' in n:
                break
        print(time.time() - tick)

    for dataset in ["wikitext2", "c4"]:
        dataloader, testloader = get_loaders(
            dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
        )
        print("Dataset:", dataset)
        qwen_eval_ppl(model, testloader, DEV, dataset, args.log_wandb)

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # if args.distribute_model == "True":
    #     distribute_model(model)
    # else:
    #     model.to(DEV)

    # evaluate_zero_shot(model, tokenizer, batch_size=8)

    if args.save:
        torch.save(model.state_dict(), args.save)
