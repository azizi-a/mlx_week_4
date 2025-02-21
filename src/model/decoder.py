import torch
import torch.nn as nn
from .attention import Attention
from config import EMBEDDING_DIM, DECODER_ATTENTION_BLOCK_COUNT


class Decoder(nn.Module):
  def __init__(self, vocab_size):
    super(Decoder, self).__init__()

    self.embedding = nn.Embedding(vocab_size, EMBEDDING_DIM)
    self.norm_0 = nn.LayerNorm(EMBEDDING_DIM)
    self.norm_1 = nn.LayerNorm(EMBEDDING_DIM)
    self.self_attention = Attention()
    self.self_attention_blocks = nn.ModuleList([Attention() for _ in range(DECODER_ATTENTION_BLOCK_COUNT)])
    self.norm_2 = nn.LayerNorm(EMBEDDING_DIM)
    self.output_layer = nn.Linear(EMBEDDING_DIM, vocab_size)
    self.softmax = nn.Softmax(dim=1)

  def forward(self, caption, image):
    caption = self.embedding(caption)
    caption = torch.cat([image.unsqueeze(1), caption], dim=1)
    caption = self.norm_1(caption)
    image = self.norm_1(image)
    size = caption.size(1)
    mask = torch.triu(torch.ones(size, size), diagonal=1)
    for self_attention in self.self_attention_blocks:
      caption, _ = self_attention(caption, caption, mask)
    caption = self.norm_2(caption)
    caption = self.output_layer(caption)
    caption = self.softmax(caption)
    return caption
