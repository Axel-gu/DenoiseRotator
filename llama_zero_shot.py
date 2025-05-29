from lm_eval_utils import evaluate_zero_shot
from llama_utils import load_rotated_llama
from gpu_utils import distribute_model
from llama_prune import llama_eval_ppl

if __name__ == "__main__":

    import argparse
    from data_utils import *

    parser = argparse.ArgumentParser()

    parser.add_argument("--config_path", type=str, help="LlaMA config to load")
    parser.add_argument("--state_dict_path", type=str, help="State dict to load")
    parser.add_argument("--distribute_model", type=str, default="False")
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed for sampling the calibration data."
    )
    parser.add_argument(
        "--log_wandb", action="store_true", help="Whether to log to wandb."
    )

    args = parser.parse_args()

    DEV = torch.device("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(args.config_path)

    model = load_rotated_llama(args.config_path, args.state_dict_path)
    model.seqlen = 2048
    model.eval()

    for n, p in model.named_parameters():
        print(n, torch.mean((p == 0).float()))
        if 'down_proj' in n:
            break

    print(model)

    if args.distribute_model == "True":
        # distribute model across available GPUs
        distribute_model(model)
    else:
        model.to(DEV)

    evaluate_zero_shot(model, tokenizer, batch_size=8)
