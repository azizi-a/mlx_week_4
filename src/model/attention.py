import torch
import torch.nn as nn
from config import EMBEDDING_DIM, LATENT_DIM, ATTENTION_HEAD_COUNT


class Attention(nn.Module):
  def __init__(self):
    super().__init__()

    self.embedding_dim = EMBEDDING_DIM
    self.latent_dim = LATENT_DIM
    self.head_count = ATTENTION_HEAD_COUNT
    self.head_dim = LATENT_DIM // ATTENTION_HEAD_COUNT
    assert self.head_dim * ATTENTION_HEAD_COUNT == LATENT_DIM, "LATENT_DIM must be divisible by ATTENTION_HEAD_COUNT"

    self.norm_1 = nn.LayerNorm(EMBEDDING_DIM)
    self.norm_2 = nn.LayerNorm(EMBEDDING_DIM)
    self.query = nn.Linear(EMBEDDING_DIM, LATENT_DIM)
    self.key = nn.Linear(EMBEDDING_DIM, LATENT_DIM)
    self.value = nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM * ATTENTION_HEAD_COUNT)
    self.norm_3 = nn.LayerNorm(EMBEDDING_DIM)
    self.linear_in = nn.Linear(EMBEDDING_DIM, LATENT_DIM)
    self.relu = nn.ReLU()
    self.linear_out = nn.Linear(LATENT_DIM, EMBEDDING_DIM)

  def forward(self, x, y, mask=None):
    # Attention mechanism
    x = self.norm_1(x)
    y = self.norm_2(y)
    q = self.query(x).view(x.size(0), -1, self.head_count, self.head_dim).transpose(1, 2)
    k = self.key(y).view(y.size(0), -1, self.head_count, self.head_dim).transpose(1, 2)
    v = self.value(y).view(y.size(0), -1, self.head_count, self.embedding_dim).transpose(1, 2)
    a = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
    # mask is 0 for valid positions, 1 for invalid positions
    if mask is not None:
      a = a.masked_fill(mask != 0, float("-inf"))
    a = torch.softmax(a, dim=-1)
    dx = torch.matmul(a, v)
    dx = dx.sum(dim=1)
    x = x + dx
    x = self.norm_3(x)

    # MLP
    dx = self.linear_in(x)
    dx = self.relu(dx)
    dx = self.linear_out(dx)
    x = x + dx
    return x, y
