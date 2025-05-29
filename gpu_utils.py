import logging
import time

import numpy as np
import torch
from accelerate import dispatch_model, infer_auto_device_map
from accelerate.utils import get_balanced_memory
from torch.utils.data import DataLoader
from tqdm import tqdm

import gc

def distribute_model(model) -> None:
    """Distribute the model across available GPUs."""
    max_memory = get_balanced_memory(
        model, no_split_module_classes=["LlamaDecoderLayer", "RotatedLlamaDecoderLayer", "Qwen2DecoderLayer", "RotatedQwenDecoderLayer"]
    )

    device_map = infer_auto_device_map(
        model, max_memory=max_memory, no_split_module_classes=["LlamaDecoderLayer", "RotatedLlamaDecoderLayer", "Qwen2DecoderLayer", "RotatedQwenDecoderLayer"]
    )

    dispatch_model(
        model,
        device_map=device_map,
        offload_buffers=True,
        offload_dir="offload",
        state_dict=model.state_dict(),
    )

    gc.collect()


def sync_gpus() -> None:
    """Sync all GPUs to make sure all operations are finished, needed for correct benchmarking of latency/throughput."""
    for i in range(torch.cuda.device_count()):
        torch.cuda.synchronize(device=i)
