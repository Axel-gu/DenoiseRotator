import math
import time

import torch
import torch.nn as nn
import transformers

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

## SparseGPT: https://github.com/IST-DASLab/sparsegpt/tree/f5c25005a61f96a0933ca2f95705a963585aafaa
class SparseGPT:

    def __init__(self, layer):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0

    def add_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear) or isinstance(self.layer, transformers.Conv1D):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())

    def sparsegpt_prune(
        self, sparsity, prune_n=0, prune_m=0, blocksize=128, percdamp=.01
    ):
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        tick = time.time()

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        Losses = torch.zeros(self.rows, device=self.dev)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        mask = None

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            if prune_n == 0: 
                if mask is not None:
                    mask1 = mask[:, i1:i2]
                else:
                    tmp = W1 ** 2 / (torch.diag(Hinv1).reshape((1, -1))) ** 2
                    thresh = torch.sort(tmp.flatten())[0][int(tmp.numel() * sparsity)]
                    mask1 = tmp <= thresh
            else:
                mask1 = torch.zeros_like(W1) == 1

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if prune_n != 0 and i % prune_m == 0:
                    tmp = W1[:, i:(i + prune_m)] ** 2 / (torch.diag(Hinv1)[i:(i + prune_m)].reshape((1, -1))) ** 2
                    mask1.scatter_(1, i + torch.topk(tmp, prune_n, dim=1, largest=False)[1], True)

                q = w.clone()
                q[mask1[:, i]] = 0

                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d ** 2

                err1 = (w - q) / d 
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            W[:, i1:i2] = Q1
            Losses += torch.sum(Losses1, 1) / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        torch.cuda.synchronize()
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.layer.weight.data = W.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)

    def wanda_prune(self, sparsity, prune_n=0, prune_m=0, all=False):
        W_metric = torch.abs(self.layer.weight.data) @ torch.diag(torch.sqrt(torch.diag(self.H))).to(self.layer.weight.data.dtype)
        W_mask = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False
        if prune_n != 0:
            # semi-structured n:m sparsity
            for ii in range(W_metric.shape[1]):
                if ii % prune_m == 0:
                    tmp = W_metric[:,ii:(ii+prune_m)].float()
                    W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
        else:
            # unstructured
            if all:
                W_metric_flattened = W_metric.flatten()
                sort_indices = torch.argsort(W_metric_flattened, stable=True)
                
                # Calculate the number of elements to mask based on sparsity
                num_elements_to_mask = int(W_metric_flattened.size(0) * sparsity)
                masked_indices_flattened = sort_indices[:num_elements_to_mask]
                
                # Create a mask from the flattened indices
                W_mask_flattened = torch.zeros_like(W_metric_flattened, dtype=torch.bool)
                W_mask_flattened[masked_indices_flattened] = True
                W_mask = W_mask_flattened.view(W_metric.size())
            else:
                sort_res = torch.sort(W_metric, dim=-1, stable=True)
                indices = sort_res[1][:,:int(W_metric.shape[1]*sparsity)]
                W_mask.scatter_(1, indices, True)
        self.layer.weight.data[W_mask] = 0

    def magnitude_prune(self, sparsity, prune_n=0, prune_m=0, all=True):
        W_metric = torch.abs(self.layer.weight.data)
        W_mask = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False
        if prune_n != 0:
            # semi-structured n:m sparsity
            for ii in range(W_metric.shape[1]):
                if ii % prune_m == 0:
                    tmp = W_metric[:,ii:(ii+prune_m)].float()
                    W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
        else:
            # unstructured
            if all:
                W_metric_flattened = W_metric.flatten()
                sort_indices = torch.argsort(W_metric_flattened, stable=True)
                
                # Calculate the number of elements to mask based on sparsity
                num_elements_to_mask = int(W_metric_flattened.size(0) * sparsity)
                masked_indices_flattened = sort_indices[:num_elements_to_mask]
                
                # Create a mask from the flattened indices
                W_mask_flattened = torch.zeros_like(W_metric_flattened, dtype=torch.bool)
                W_mask_flattened[masked_indices_flattened] = True
                W_mask = W_mask_flattened.view(W_metric.size())
            else:
                sort_res = torch.sort(W_metric, dim=-1, stable=True)
                indices = sort_res[1][:,:int(W_metric.shape[1]*sparsity)]
                W_mask.scatter_(1, indices, True)
        self.layer.weight.data[W_mask] = 0

    def reset_hessian(self):
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0
        torch.cuda.empty_cache()

    def free(self):
        self.H = None
        torch.cuda.empty_cache()