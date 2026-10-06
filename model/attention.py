import torch
import torch.nn as nn
from torchtyping import TensorType

class SingleHeadAttention(nn.Module):

    def __init__(self, embedding_dim: int, attention_dim: int):
        super().__init__()
        torch.manual_seed(0)
        # Create three linear projections (Key, Query, Value) with bias=False
        # Instantiation order matters for reproducible weights: key, query, value
        self.key = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.query = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.value = nn.Linear(embedding_dim, attention_dim, bias=False)

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        # 1. Project input through K, Q, V linear layers
        # 2. Compute attention scores: (Q @ K^T) / sqrt(attention_dim)
        # 3. Apply causal mask: use torch.tril(torch.ones(...)) to build lower-triangular matrix,
        #    then masked_fill positions where mask == 0 with float('-inf')
        # 4. Apply softmax(dim=2) to masked scores
        # 5. Return (scores @ V) rounded to 4 decimal places
        # 1. Project input through K, Q, V linear layers
        Q = self.query(embedded)
        K = self.key(embedded)
        V = self.value(embedded)

        # 2. Compute scaled attention scores: (Q @ K^T) / sqrt(attention_dim)
        # K is transposed along the last two dimensions: (seq_len, attention_dim) -> (attention_dim, seq_len)
        attention_dim = Q.shape[-1]
        scores = (Q @ K.transpose(-2, -1)) / (attention_dim ** 0.5)

        # 3. Apply causal mask
        seq_len = embedded.shape[1]
        mask = torch.tril(torch.ones(seq_len, seq_len, device=embedded.device))
        scores = scores.masked_fill(mask == 0, float('-inf'))

        # 4. Apply softmax along the last dimension (keys dimension)
        attention_weights = torch.softmax(scores, dim=2)

        # 5. Compute weighted sum of values and round to 4 decimal places
        output = attention_weights @ V
        return torch.round(output, decimals=4)

