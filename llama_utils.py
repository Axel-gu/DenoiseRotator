import numpy as np
import torch
from torch import nn
from transformers import AutoModelForCausalLM
import sys
import time
import os
from torch.cuda.amp import autocast

from torch import matmul
from transformers.models.llama.modeling_llama import LlamaConfig, LlamaDecoderLayer, LlamaForCausalLM, LlamaRMSNorm
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.processing_utils import Unpack
from typing import Callable, List, Optional, Tuple, Union

from transformers import AutoConfig, AutoModelForCausalLM
from accelerate import init_empty_weights

def get_llama(model):
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

def llama_fuse_rms_single_layer(layer):
    layer.self_attn.q_proj.weight.data = layer.self_attn.q_proj.weight.data @ torch.diag(layer.input_layernorm.weight.data)
    layer.self_attn.k_proj.weight.data = layer.self_attn.k_proj.weight.data @ torch.diag(layer.input_layernorm.weight.data)
    layer.self_attn.v_proj.weight.data = layer.self_attn.v_proj.weight.data @ torch.diag(layer.input_layernorm.weight.data)
    layer.input_layernorm.weight.data = torch.ones_like(layer.input_layernorm.weight.data, dtype=layer.input_layernorm.weight.dtype, device=layer.input_layernorm.weight.device)

    layer.mlp.up_proj.weight.data = layer.mlp.up_proj.weight.data @ torch.diag(layer.post_attention_layernorm.weight.data)
    layer.mlp.gate_proj.weight.data = layer.mlp.gate_proj.weight.data @ torch.diag(layer.post_attention_layernorm.weight.data)
    layer.post_attention_layernorm.weight.data = torch.ones_like(layer.post_attention_layernorm.weight.data, dtype=layer.post_attention_layernorm.weight.dtype, device=layer.post_attention_layernorm.weight.device)

@torch.compile
def row_entropy_sum(matrix):
    # matrix = matrix.to(torch.float32)
    abs_sq = torch.nan_to_num(matrix, nan=0.0, posinf=1e5, neginf=0)
    row_sums = torch.sum(abs_sq, dim=1, keepdim=True)
    row_sums = torch.where(row_sums == 0, torch.ones_like(row_sums), row_sums)
    probs = abs_sq / row_sums
    probs = torch.nan_to_num(probs, nan=0.0, posinf=1e5, neginf=0)
    
    probs = torch.where(probs > 0, probs, 1)
    log_probs = torch.log(probs)
    log_probs = torch.nan_to_num(log_probs, nan=0.0, posinf=1e5, neginf=0)
    
    entropies = -torch.sum(probs * log_probs, dim=1)
    res = torch.sum(entropies)
    if torch.isnan(res).any():
        print("WARNING")
    res = torch.nan_to_num(res, nan=0.0, posinf=0, neginf=0)
    return res

class RotatorOptimizer(torch.nn.Module):
    def __init__(self, weight_dict_list, r_dim, num_key_value_heads, head_dim, device, positive=True, hessian_dict_list=None, num_piece = 1):
        super().__init__()

        self.weight_dict_list = weight_dict_list
        self.num_piece = num_piece
        self.r_dim = r_dim
        self.device = device
        self.A_dim = self.r_dim // self.num_piece
        self.A_list = [torch.nn.Parameter(torch.eye(self.A_dim, device=device)) for i in range(self.num_piece)]
        self.positive = positive
        self.hessian_dict_list = hessian_dict_list
        self.num_layer = len(weight_dict_list)
        self.B_list_list = []
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.dtype = torch.bfloat16
        for i in range(self.num_layer):
            self.B_list_list.append([torch.nn.Parameter(torch.eye(self.head_dim, device=device)) for j in range(self.num_key_value_heads)])
        
        for idx in range(self.num_layer):
            for name in self.weight_dict_list[idx]:
                self.weight_dict_list[idx][name] = self.weight_dict_list[idx][name].weight.detach().to(self.device).to(self.dtype)
                self.weight_dict_list[idx][name].requires_grad_(False)
            for name in self.hessian_dict_list[idx]:
                self.hessian_dict_list[idx][name] = self.hessian_dict_list[idx][name].detach().to(self.device).to(self.dtype)
                self.hessian_dict_list[idx][name].requires_grad_(False)

    def parameters(self, recurse=True):
        res = []
        for l in self.A_list:
            res.append(l)
        for l in self.B_list_list:
            res += l
        return res

    def get_orthogonal_matrix(self):
        Q = torch.block_diag(*[torch.linalg.qr(self.A_list[i])[0] for i in range(self.num_piece)]).to(dtype=self.dtype)
        return Q
    
    def get_orthogonal_matrix_R2_list_list(self):
        R2_list_list = []
        for i in range(self.num_layer):
            R2_list = []
            for j in range(self.num_key_value_heads):
                R2_list.append(torch.linalg.qr(self.B_list_list[i][j])[0].to(dtype=self.dtype))
            R2_list_list.append(R2_list)
        return R2_list_list

    def get_R1_list(self):
        return [torch.linalg.qr(self.A_list[i])[0] for i in range(self.num_piece)]

    def get_R2_list_list(self):
        return self.get_orthogonal_matrix_R2_list_list()
    
    def compute_salience_RWX(self, weight, hessian, R):
        raise NotImplementedError

    def compute_salience_WR_1RX(self, weight, hessian, R):
        raise NotImplementedError

    def compute_salience_R2WR_1RX(self, weight, hessian, R, R2_list):
        raise NotImplementedError

    def compute_salience_RWR2_1R2X(self, weight, hessian, R, R2_list):
        raise NotImplementedError

    def forward(self, indices_dict=None):
        R = self.get_orthogonal_matrix()

        R2_list_list = self.get_orthogonal_matrix_R2_list_list()
        for idx in range(len(R2_list_list)):
            for j in range(len(R2_list_list[idx])):
                R2_list_list[idx][j] = R2_list_list[idx][j]
        loss = None

        WR_1RX_list = ["self_attn.q_proj", "self_attn.k_proj", "mlp.gate_proj", "mlp.up_proj"]
        RWX_list = ["mlp.down_proj"]
        R2WR_1RX_list = ["self_attn.v_proj"]
        RWR2_1R2X_list = ["self_attn.o_proj"]

        hidden_list = ["self_attn.q_proj", "self_attn.o_proj"]
        intermediate_list = ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]

        for idx in range(self.num_layer):
            for name in RWX_list:
                weight = self.weight_dict_list[idx][name]
                hessian = self.hessian_dict_list[idx][name]
                if indices_dict != None:
                    if name in hidden_list:
                        weight = weight[:, indices_dict["hidden"]]
                        hessian = hessian[indices_dict["hidden"], :][:, indices_dict["hidden"]]
                    elif name in intermediate_list:
                        weight = weight[:, indices_dict["intermediate"]]
                        hessian = hessian[indices_dict["intermediate"], :][:, indices_dict["intermediate"]]                
                
                salience = self.compute_salience_RWX(weight, hessian, R)

                layer_loss = row_entropy_sum(salience.T)

                if loss == None:
                    loss = layer_loss
                else:
                    loss += layer_loss

            for name in WR_1RX_list:
                weight = self.weight_dict_list[idx][name]
                if indices_dict != None:
                    if name in hidden_list:
                        weight = weight[indices_dict["hidden"], :]
                    elif name in intermediate_list:
                        weight = weight[indices_dict["intermediate"], :]
                hessian = self.hessian_dict_list[idx][name]
                salience = self.compute_salience_WR_1RX(weight, hessian, R)

                layer_loss = row_entropy_sum(salience)

                if loss == None:
                    loss = layer_loss
                else:
                    loss += layer_loss

            for name in R2WR_1RX_list:
                weight = self.weight_dict_list[idx][name]
                hessian = self.hessian_dict_list[idx][name]
                salience = self.compute_salience_R2WR_1RX(weight, hessian, R, R2_list_list[idx])

                layer_loss = row_entropy_sum(salience) + row_entropy_sum(salience.T)

                if loss == None:
                    loss = layer_loss
                else:
                    loss += layer_loss

            for name in RWR2_1R2X_list:
                weight = self.weight_dict_list[idx][name]
                hessian = self.hessian_dict_list[idx][name]
                salience = self.compute_salience_RWR2_1R2X(weight, hessian, R, R2_list_list[idx])

                layer_loss = row_entropy_sum(salience) + row_entropy_sum(salience.T)

                if loss == None:
                    loss = layer_loss
                else:
                    loss += layer_loss

        if self.positive:
            return loss
        else:
            return -loss

@torch.compile
def compute_salience_RWX_wanda(weight, hessian, R):
        x_norms = torch.abs(torch.diag(hessian))
        wanda_salience = ((R.T @ weight) ** 2) * x_norms 
        return wanda_salience

@torch.compile
def compute_salience_WR_1RX_wanda(weight, hessian, R):
    rotated_hessian = R.T @ hessian @ R
    x_norms = torch.abs(torch.diag(rotated_hessian))
    wanda_salience = ((weight @ R) ** 2) * x_norms
    return wanda_salience

@torch.compile
def compute_salience_R2WR_1RX_wanda(weight, hessian, R, R2_list):
    R2 = torch.block_diag(*R2_list)
    rotated_hessian = R.T @ hessian @ R
    x_norms = torch.abs(torch.diag(rotated_hessian))
    wanda_salience = ((R2.T @ weight @ R) ** 2) * x_norms
    return wanda_salience

@torch.compile
def compute_salience_RWR2_1R2X_wanda(weight, hessian, R, R2_list):
    r2_list = []
    for r2 in R2_list:
        for i in range(weight.shape[1] // r2.shape[0] // len(R2_list)):
            r2_list.append(r2)
    R2 = torch.block_diag(*r2_list)
    rotated_hessian = R2.T @ hessian @ R2
    x_norms = torch.abs(torch.diag(rotated_hessian))
    wanda_salience = ((R.T @ weight @ R2) ** 2) * x_norms
    return wanda_salience

class RotatorOptimizer_wanda(RotatorOptimizer):

    def compute_salience_RWX(self, weight, hessian, R):
        return compute_salience_RWX_wanda(weight, hessian, R)

    def compute_salience_WR_1RX(self, weight, hessian, R):
        return compute_salience_WR_1RX_wanda(weight, hessian, R)

    def compute_salience_R2WR_1RX(self, weight, hessian, R, R2_list):
        return compute_salience_R2WR_1RX_wanda(weight, hessian, R, R2_list)

    def compute_salience_RWR2_1R2X(self, weight, hessian, R, R2_list):
        return compute_salience_RWR2_1R2X_wanda(weight, hessian, R, R2_list)
    


class RotatorOptimizer_magnitude(RotatorOptimizer):
    
    def compute_salience_RWX(self, weight, hessian, R):
        magnitude_salience = (R.T @ weight) ** 2
        return magnitude_salience

    def compute_salience_WR_1RX(self, weight, hessian, R):
        magnitude_salience = (weight @ R) ** 2
        return magnitude_salience

    def compute_salience_R2WR_1RX(self, weight, hessian, R, R2_list):
        R2 = torch.block_diag(*R2_list)
        magnitude_salience = (R2.T @ weight @ R) ** 2
        return magnitude_salience

    def compute_salience_RWR2_1R2X(self, weight, hessian, R, R2_list):
        r2_list = []
        for r2 in R2_list:
            for i in range(weight.shape[1] // r2.shape[0] // len(R2_list)):
                r2_list.append(r2)
        R2 = torch.block_diag(*r2_list)
        magnitude_salience = (R.T @ weight @ R2) ** 2
        return magnitude_salience

@torch.compile 
def compute_salience_RWX_sparsegpt(weight, hessian, R):
    hinv_diag = torch.abs(torch.diag(hessian))
    sparsegpt_salience = ((R.T @ weight) ** 2) / hinv_diag 
    return sparsegpt_salience

@torch.compile 
def compute_salience_WR_1RX_sparsegpt(weight, hessian, R):
    hinv = hessian
    rotated_hinv = R.T @ hinv @ R
    hinv_diag = torch.abs(torch.diag(rotated_hinv))
    sparsegpt_salience = ((weight @ R) ** 2) / hinv_diag
    return sparsegpt_salience

@torch.compile 
def compute_salience_R2WR_1RX_sparsegpt(weight, hessian, R, R2_list):
    R2 = torch.block_diag(*R2_list)
    hinv = hessian
    rotated_hinv = R.T @ hinv @ R
    hinv_diag = torch.abs(torch.diag(rotated_hinv))
    sparsegpt_salience = ((R2.T @ weight @ R) ** 2) / hinv_diag
    return sparsegpt_salience

@torch.compile 
def compute_salience_RWR2_1R2X_sparsegpt(weight, hessian, R, R2_list):
    hinv = hessian
    r2_list = []
    for r2 in R2_list:
        for i in range(weight.shape[1] // r2.shape[0] // len(R2_list)):
            r2_list.append(r2)
    R2 = torch.block_diag(*r2_list)
    rotated_hinv = R2.T @ hinv @ R2
    hinv_diag = torch.abs(torch.diag(rotated_hinv))
    sparsegpt_salience = ((R.T @ weight @ R2) ** 2) / hinv_diag
    return sparsegpt_salience


class RotatorOptimizer_sparsegpt(RotatorOptimizer):
    
    def __init__(self, weight_dict_list, r_dim, num_key_value_heads, head_dim, device, positive=True, hessian_dict_list=None, num_piece = 1, percdamp=.01):
        super().__init__(weight_dict_list, r_dim, num_key_value_heads, head_dim, device, positive=positive, hessian_dict_list=hessian_dict_list, num_piece = num_piece)
        self.inverse_hessian(percdamp=percdamp)

    def inverse_hessian(self, percdamp=.01):
        hinv_dict_list = []
        for idx in range(self.num_layer):
            hinv_dict_list.append({})
            for name in self.hessian_dict_list[idx]:
                H = self.hessian_dict_list[idx][name].to(dtype=torch.float32)
                dead = torch.diag(H) == 0
                H[dead, dead] = 1
                damp = percdamp * torch.mean(torch.diag(H))
                diag = torch.arange(H.shape[0], device=H.device)
                H[diag, diag] += damp

                success = False
                attemps = 0
                while not success:
                    try:
                        H = torch.inverse(H)
                        success = True
                    except RuntimeError as e:
                        print(f"Attempt {attemps}: Matrix not positive definite, modifying diagonal elements.")
                    H[diag, diag] += damp
                    attemps += 1

                hinv_dict_list[idx][name] = H.to(dtype=torch.bfloat16)
        self.hessian_dict_list = hinv_dict_list
        torch.cuda.empty_cache()

    def compute_salience_RWX(self, weight, hessian, R):
        return compute_salience_RWX_sparsegpt(weight, hessian, R)

    def compute_salience_WR_1RX(self, weight, hessian, R):
        return compute_salience_WR_1RX_sparsegpt(weight, hessian, R)

    def compute_salience_R2WR_1RX(self, weight, hessian, R, R2_list):
        return compute_salience_R2WR_1RX_sparsegpt(weight, hessian, R, R2_list)

    def compute_salience_RWR2_1R2X(self, weight, hessian, R, R2_list):
        return compute_salience_RWR2_1R2X_sparsegpt(weight, hessian, R, R2_list)

def llama_fuse_rotation_single_layer(layer, R1, R2_list):
    R2_list_o = []
    for r2 in R2_list:
        for i in range(layer.self_attn.v_proj.weight.data.shape[1] // layer.self_attn.v_proj.weight.data.shape[0]):
            R2_list_o.append(r2)
    R2_transform_o = torch.block_diag(*R2_list_o).to(R1.device)
    R2_transform_v = torch.block_diag(*R2_list).to(R1.device)
    layer.self_attn.q_proj.weight.data = layer.self_attn.q_proj.weight.data @ R1.T
    layer.self_attn.k_proj.weight.data = layer.self_attn.k_proj.weight.data @ R1.T
    layer.self_attn.v_proj.weight.data = R2_transform_v.T @ layer.self_attn.v_proj.weight.data @ R1.T
    layer.self_attn.o_proj.weight.data = R1 @ layer.self_attn.o_proj.weight.data @ R2_transform_o
    layer.mlp.up_proj.weight.data = layer.mlp.up_proj.weight.data @ R1.T
    layer.mlp.gate_proj.weight.data = layer.mlp.gate_proj.weight.data @ R1.T
    layer.mlp.down_proj.weight.data = R1 @ layer.mlp.down_proj.weight.data

class RotatedLlamaDecoderLayer(LlamaDecoderLayer):

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        
        # hidden_states (bsz, length, hidden_dim)
        if self.R1 is not None:
            hidden_states = matmul(hidden_states, self.R1.T)
        
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        if self.R1 is not None:
            hidden_states = matmul(hidden_states, self.R1)

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs

def replace_llama_layer(model, layer_idx, R1):

    rotated_layer = RotatedLlamaDecoderLayer(model.config, layer_idx).to(model.config.torch_dtype)
    rotated_layer.load_state_dict(model.model.layers[layer_idx].state_dict(), strict=True)

    hidden_size = model.config.hidden_size
    rotated_layer.R1 = torch.nn.Parameter(R1.to(model.config.torch_dtype).to(model.device))
    model.model.layers[layer_idx] = rotated_layer

def load_rotated_llama(config_path, state_dict_path):
    config = AutoConfig.from_pretrained(config_path)
    model = AutoModelForCausalLM.from_config(config)
    
    hidden_size = config.hidden_size
    for idx in range(config.num_hidden_layers):
        R1 = torch.eye(hidden_size)
        replace_llama_layer(model, idx, R1)
    
    state_dict = torch.load(state_dict_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    
    return model

def load_rotated_llama_fast(config_path, state_dict_path):
    model = get_llama(config_path)
    
    hidden_size = model.config.hidden_size
    for idx in range(model.config.num_hidden_layers):
        R1 = torch.eye(hidden_size)
        replace_llama_layer(model, idx, R1)
    
    state_dict = torch.load(state_dict_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    
    return model

@torch.no_grad()
def llama_eval_ppl_layer_difference(model, loaded_model, testenc, dev, dataset: str, log_wandb: bool = False):
    print("Evaluating ...")

    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    loaded_model.config.use_cache = False
    layers = model.model.layers
    loaded_layers = loaded_model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)

    model.model.rotary_emb = model.model.rotary_emb.to(dev)

    loaded_model.model.embed_tokens = loaded_model.model.embed_tokens.to(dev)

    loaded_model.model.rotary_emb = loaded_model.model.rotary_emb.to(dev)

    layers[0] = layers[0].to(dev)
    loaded_layers[0] = loaded_layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0, "attention_mask": None}

    loaded_inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    loaded_cache = {"i": 0, "attention_mask": None}

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
    
    class loaded_Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            loaded_inps[loaded_cache["i"]] = inp
            loaded_cache["i"] += 1
            loaded_cache["attention_mask"] = kwargs["attention_mask"]
            loaded_cache["position_embeddings"] = kwargs["position_embeddings"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    loaded_layers[0] = loaded_Catcher(loaded_layers[0])
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
        try:
            loaded_model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module
    loaded_layers[0] = loaded_layers[0].module

    layers[0] = layers[0].cpu()
    loaded_layers[0] = loaded_layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    loaded_model.model.embed_tokens = loaded_model.model.embed_tokens
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    loaded_outs = torch.zeros_like(loaded_inps)
    attention_mask = cache["attention_mask"]
    position_embeddings = cache["position_embeddings"]

    for i in range(len(layers)):
        print(i)

        layer = layers[i].to(dev)
        loaded_layer = loaded_layers[i].to(dev)

        layer.load_state_dict(loaded_layer.state_dict())

        print(model.config.hidden_act, loaded_model.config.hidden_act)
        print("R1: ", torch.sum(torch.abs(layer.R1.data - loaded_layer.R1.data)))
        print("input_layernorm: ", torch.sum(torch.abs(layer.input_layernorm.weight.data - loaded_layer.input_layernorm.weight.data)))
        print("q_proj: ", torch.sum(torch.abs(layer.self_attn.q_proj.weight.data - loaded_layer.self_attn.q_proj.weight.data)))
        print("k_proj: ", torch.sum(torch.abs(layer.self_attn.k_proj.weight.data - loaded_layer.self_attn.k_proj.weight.data)))
        print("v_proj: ", torch.sum(torch.abs(layer.self_attn.v_proj.weight.data - loaded_layer.self_attn.v_proj.weight.data)))
        print("o_proj: ", torch.sum(torch.abs(layer.self_attn.o_proj.weight.data - loaded_layer.self_attn.o_proj.weight.data)))

        print("post_attention_layernorm: ", torch.sum(torch.abs(layer.post_attention_layernorm.weight.data - loaded_layer.post_attention_layernorm.weight.data)))
        print("up_proj: ", torch.sum(torch.abs(layer.mlp.up_proj.weight.data - loaded_layer.mlp.up_proj.weight.data)))
        print("gate_proj ", torch.sum(torch.abs(layer.mlp.gate_proj.weight.data - loaded_layer.mlp.gate_proj.weight.data)))
        print("down_proj ", torch.sum(torch.abs(layer.mlp.down_proj.weight.data - loaded_layer.mlp.down_proj.weight.data)))

        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_embeddings=position_embeddings)[0]
            loaded_outs[j] = loaded_layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_embeddings=position_embeddings)[0]

        print("Output Difference: ", torch.sum(torch.abs(outs-loaded_outs)))

        layers[i] = layer.cpu()
        loaded_layers[i] = loaded_layer.cpu()
        del layer
        del loaded_layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps
        loaded_inps, loaded_outs = loaded_outs, loaded_inps
