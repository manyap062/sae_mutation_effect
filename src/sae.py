"""minimal sae model for loading pretrained weights"""
import math
import torch
import torch.nn as nn
from torch.nn import functional as F


class SparseAutoencoder(nn.Module):
    def __init__(self, d_model: int = 1280, d_hidden: int = 4096, k: int = 32):
        super().__init__()
        self.w_enc = nn.Parameter(torch.empty(d_model, d_hidden))
        self.w_dec = nn.Parameter(torch.empty(d_hidden, d_model))
        self.b_enc = nn.Parameter(torch.zeros(d_hidden))
        self.b_pre = nn.Parameter(torch.zeros(d_model))
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.k = k
        nn.init.kaiming_uniform_(self.w_enc, a=math.sqrt(5))
        self.w_dec.data = self.w_enc.data.T.clone()

    def LN(self, x: torch.Tensor, eps: float = 1e-5):
        # layer normalization returning x, mu, std
        mu = x.mean(dim=-1, keepdim=True)
        x = x - mu
        std = x.std(dim=-1, keepdim=True)
        x = x / (std + eps)
        return x, mu, std

    def topK_activation(self, x: torch.Tensor, k: int) -> torch.Tensor:
        # topK ReLU activation
        topk = torch.topk(x, k=k, dim=-1, sorted=False)
        values = F.relu(topk.values)
        result = torch.zeros_like(x)
        result.scatter_(-1, topk.indices, values)
        return result

    @torch.no_grad()
    def encode(self, x: torch.Tensor):
        """
        returns pre-activations BEFORE topk + normalization stats
        this is intentional for patching — patching happens in pre-activation space
        """
        x, mu, std = self.LN(x)
        x = x - self.b_pre
        pre_acts = x @ self.w_enc + self.b_enc
        return pre_acts, mu, std

    @torch.no_grad()
    def decode(self, pre_acts: torch.Tensor, mu: torch.Tensor, std: torch.Tensor):
        """decode from pre-activations (applies topK internally)"""
        latents = self.topK_activation(pre_acts, self.k)
        recons = latents @ self.w_dec + self.b_pre
        recons = recons * std + mu
        return recons

    @torch.no_grad()
    def decode_with_patch(self, pre_acts: torch.Tensor, mu: torch.Tensor, std: torch.Tensor,
                          patch_indices: torch.Tensor, patch_values: torch.Tensor):
        """
        decode with specific feature indices patched to new values
        
        """
        pre_acts = pre_acts.clone()
        pre_acts[:, patch_indices] = patch_values
        return self.decode(pre_acts, mu, std)

    def decode_with_grad(self, pre_acts: torch.Tensor, mu: torch.Tensor, std: torch.Tensor):
        """
        decode WITH gradients (for integrated gradients)
        same as decode() but no @torch.no_grad() decorator
        """
        latents = self.topK_activation(pre_acts, self.k)
        recons = latents @ self.w_dec + self.b_pre
        recons = recons * std + mu
        return recons
