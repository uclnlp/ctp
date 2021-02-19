# -*- coding: utf-8 -*-

import torch
from torch import nn, Tensor

from typing import Tuple, Optional, List

import logging

logger = logging.getLogger(__name__)


class BatchMulti(nn.Module):
    def __init__(self,
                 models: List[nn.Module]):
        super().__init__()
        self.models = nn.ModuleList(models)
        logger.info(f'BatchHoppy(models={[m.__class__.__name__ for m in self.models]})')

    def score(self,
              rel: Tensor, arg1: Tensor, arg2: Tensor,
              facts: List[Tensor],
              nb_facts: Tensor,
              entity_embeddings: Tensor,
              nb_entities: Tensor) -> Tensor:
        res = None
        for m in self.models:
            m_scores = m.score(rel, arg1, arg2, facts, nb_facts, entity_embeddings, nb_entities)
            res = m_scores if res is None else torch.min(res, m_scores)
        assert res is not None
        return res

    def forward(self,
                rel: Tensor, arg1: Optional[Tensor], arg2: Optional[Tensor],
                facts: List[Tensor],
                nb_facts: Tensor,
                entity_embeddings: Tensor,
                nb_entities: Tensor) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        res_sp = res_po = None
        for m in self.models:
            m_res_sp, m_res_po = m.forward(rel, arg1, arg2, facts, nb_facts, entity_embeddings, nb_entities)
            res_sp = m_res_sp if res_sp is None else torch.min(res_sp, m_res_sp)
            res_po = m_res_po if res_po is None else torch.min(res_po, m_res_po)
        return res_sp, res_po

    def factor(self,
               embedding_vector: Tensor) -> Tensor:
        return self.models[0].factor(embedding_vector)

    def extra_factors(self,
                      rel: Tensor, arg1: Optional[Tensor], arg2: Optional[Tensor]) -> List[Tensor]:
        return [hop_generator(rel) for hop_generators in self.models[0].hops_lst for hop_generator in hop_generators]
