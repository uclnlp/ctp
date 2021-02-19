# -*- coding: utf-8 -*-

import torch
from torch import nn, Tensor

from ctp.models.base import BaseLatentFeatureModel

from typing import Tuple, Optional

import logging

logger = logging.getLogger(__name__)


class DistMult(BaseLatentFeatureModel):
    def __init__(self,
                 entity_embeddings: nn.Embedding) -> None:
        super().__init__()
        self.entity_embeddings = entity_embeddings

    def score(self,
              rel: Tensor,
              arg1: Tensor,
              arg2: Tensor,
              *args, **kwargs) -> Tensor:
        # [B]
        res = torch.sum(rel * arg1 * arg2, 1)
        return res

    def forward(self,
                rel: Tensor,
                arg1: Optional[Tensor],
                arg2: Optional[Tensor],
                *args, **kwargs) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        # [N, E]
        emb = self.entity_embeddings.weight

        # [B, N] = [B, E] @ [E, N]
        score_sp = (rel * arg1) @ emb.t() if arg1 is not None else None
        score_po = (rel * arg2) @ emb.t() if arg2 is not None else None
        return score_sp, score_po

    def factor(self,
               embedding_vector: Tensor) -> Tensor:
        return embedding_vector
