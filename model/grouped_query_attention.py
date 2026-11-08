import torch
import torch.nn as nn
from torchtyping import TensorType

class GroupedQueryAttention(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, num_kv_heads: int):
        super().__init__()
        torch.manual_seed(0)
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = model_dim // num_heads

        self.q_proj = nn.Linear(model_dim, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(model_dim, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(model_dim, num_kv_heads * self.head_dim, bias=False)
        self.output_proj = nn.Linear(num_heads * self.head_dim, model_dim, bias=False)

    def forward(self, x: TensorType[float]) -> TensorType[float]:
        B, T, D = x.shape

        # 1. Project x into Q, K, V using the projection layers
        # 2. Reshape into heads: Q has num_heads, K and V have num_kv_heads
        # 3. Expand K, V by repeating each KV head (num_heads // num_kv_heads) times
        # 4. Compute scaled dot-product attention with causal mask
        # 5. Concatenate heads and apply output projection
        # 6. Return rounded output (decimals=4)
        q = self.q_proj(x)  # (B, T, num_heads * head_dim)
        k = self.k_proj(x)  # (B, T, num_kv_heads * head_dim)
        v = self.v_proj(x)

        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)      # (B, H, T, hd)
        k = k.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)   # (B, G, T, hd)
        v = v.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        group = self.num_heads // self.num_kv_heads
        k = k.repeat_interleave(group, dim=1)  # (B, H, T, hd)
        v = v.repeat_interleave(group, dim=1)

        scores = q @ k.transpose(-2, -1) / (self.head_dim ** 0.5)  # (B, H, T, T)
        mask = torch.tril(torch.ones(T, T, device=x.device)) == 0
        scores = scores.masked_fill(mask, float('-inf'))
        weights = torch.softmax(scores, dim=-1)
        out = weights @ v  # (B, H, T, hd)

        out = out.transpose(1, 2).contiguous().view(B, T, self.num_heads * self.head_dim)
        out = self.output_proj(out)
        return torch.round(out, decimals=4)
