# -*- coding: utf-8 -*-

import logging

import torch
from torch import nn, Tensor

from ctp.models.base import BaseLatentFeatureModel

from typing import Tuple, Optional

logger = logging.getLogger(__name__)


class ComplEx(BaseLatentFeatureModel):
    def __init__(self,
                 entity_embeddings: nn.Embedding) -> None:
        super().__init__()
        self.entity_embeddings = entity_embeddings
        self.embedding_size = self.entity_embeddings.weight.shape[1] // 2

    def score(self,
              rel: Tensor,
              arg1: Tensor,
              arg2: Tensor,
              *args, **kwargs) -> Tensor:
        # [B, E]
        rel_real, rel_img = rel[:, :self.embedding_size], rel[:, self.embedding_size:]
        arg1_real, arg1_img = arg1[:, :self.embedding_size], arg1[:, self.embedding_size:]
        arg2_real, arg2_img = arg2[:, :self.embedding_size], arg2[:, self.embedding_size:]

        # [B] Tensor
        score1 = torch.sum(rel_real * arg1_real * arg2_real, 1)
        score2 = torch.sum(rel_real * arg1_img * arg2_img, 1)
        score3 = torch.sum(rel_img * arg1_real * arg2_img, 1)
        score4 = torch.sum(rel_img * arg1_img * arg2_real, 1)

        res = score1 + score2 + score3 - score4

        # [B] Tensor
        return res

    def forward(self,
                rel: Tensor,
                arg1: Optional[Tensor],
                arg2: Optional[Tensor],
                *args, **kwargs) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        emb = self.entity_embeddings.weight

        rel_real, rel_img = rel[:, :self.embedding_size], rel[:, self.embedding_size:]
        emb_real, emb_img = emb[:, :self.embedding_size], emb[:, self.embedding_size:]

        # [B] Tensor
        score_sp = score_po = None

        if arg1 is not None:
            arg1_real, arg1_img = arg1[:, :self.embedding_size], arg1[:, self.embedding_size:]

            score1_sp = (rel_real * arg1_real) @ emb_real.t()
            score2_sp = (rel_real * arg1_img) @ emb_img.t()
            score3_sp = (rel_img * arg1_real) @ emb_img.t()
            score4_sp = (rel_img * arg1_img) @ emb_real.t()

            score_sp = score1_sp + score2_sp + score3_sp - score4_sp

        if arg2 is not None:
            arg2_real, arg2_img = arg2[:, :self.embedding_size], arg2[:, self.embedding_size:]

            score1_po = (rel_real * arg2_real) @ emb_real.t()
            score2_po = (rel_real * arg2_img) @ emb_img.t()
            score3_po = (rel_img * arg2_img) @ emb_real.t()
            score4_po = (rel_img * arg2_real) @ emb_img.t()

            score_po = score1_po + score2_po + score3_po - score4_po

        return score_sp, score_po

    def factor(self,
               embedding_vector: Tensor) -> Tensor:
        vec_real = embedding_vector[:, :self.embedding_size]
        vec_img = embedding_vector[:, self.embedding_size:]
        return torch.sqrt(vec_real ** 2 + vec_img ** 2)
