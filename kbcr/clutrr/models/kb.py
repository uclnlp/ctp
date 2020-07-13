# -*- coding: utf-8 -*-

import torch
from torch import nn, Tensor

from kbcr.kernels import BaseKernel

from typing import List, Tuple, Optional

import logging

logger = logging.getLogger(__name__)


class NeuralKB(nn.Module):
    def __init__(self,
                 kernel: BaseKernel,
                 scoring_type: str = 'concat'):
        super().__init__()

        self.kernel = kernel
        self.scoring_type = scoring_type
        assert self.scoring_type in {'concat'}

    def score(self,
              rel: Tensor, arg1: Tensor, arg2: Tensor,
              facts: List[Tensor]) -> Tensor:

        fact_emb = torch.cat(facts, dim=1)
        batch_emb = torch.cat([rel, arg1, arg2], dim=1)

        # [B, KB]
        pairwise = self.kernel.pairwise(batch_emb, fact_emb)

        # [B]
        res, _ = torch.max(pairwise, dim=1)
        return res

    def forward(self,
                rel: Tensor, arg1: Optional[Tensor], arg2: Optional[Tensor],
                facts: List[Tensor],
                entity_embeddings: Tensor) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        # [N, E]
        # emb = entity_embeddings.weight
        emb = entity_embeddings

        batch_size = rel.shape[0]
        nb_entities = emb.shape[0]

        fact_emb = torch.cat(facts, dim=1)

        # [B, N, E]
        tmp = rel.view(batch_size, 1, -1).repeat(1, nb_entities, 1)

        tmp_emb = emb.view(1, nb_entities, -1).repeat(batch_size, 1, 1)

        score_sp = score_po = None

        if arg1 is not None:
            # [B, N, E]
            tmp_arg1 = arg1.view(batch_size, 1, -1).repeat(1, nb_entities, 1)
            tmp_emb_sp = torch.cat([tmp, tmp_arg1, tmp_emb], dim=2)
            # [B, N, KB]
            tmp_pairwise_sp = self.kernel.pairwise(tmp_emb_sp, fact_emb)
            score_sp, _ = torch.max(tmp_pairwise_sp, dim=2)

        if arg2 is not None:
            # [B, N, E]
            tmp_arg2 = arg2.view(batch_size, 1, -1).repeat(1, nb_entities, 1)
            tmp_emb_po = torch.cat([tmp, tmp_emb, tmp_arg2], dim=2)
            # [B, N, KB]
            tmp_pairwise_po = self.kernel.pairwise(tmp_emb_po, fact_emb)
            score_po, _ = torch.max(tmp_pairwise_po, dim=2)

        return score_sp, score_po

    def factor(self,
               embedding_vector: Tensor) -> Tensor:
        return embedding_vector
