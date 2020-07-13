# -*- coding: utf-8 -*-

import torch
from torch import nn, Tensor


class AttentiveLinear(nn.Module):
    def __init__(self,
                 embeddings: nn.Embedding):
        super().__init__()
        self.embeddings = embeddings
        nb_objects = self.embeddings.weight.shape[0]
        embedding_size = self.embeddings.weight.shape[1]
        self.projection = nn.Linear(embedding_size, nb_objects)

    def forward(self, rel: Tensor) -> Tensor:
        # [B, O]
        attn_logits = self.projection(rel)
        attn = torch.softmax(attn_logits, dim=1)
        # [B, E]
        return attn @ self.embeddings.weight
