# -*- coding: utf-8 -*-

import logging

import torch
from torch import nn, Tensor

from kbcr.models.base import BaseLatentFeatureModel

from typing import Tuple, Optional, List

logger = logging.getLogger(__name__)


class Multi(BaseLatentFeatureModel):
    def __init__(self,
                 models: List[BaseLatentFeatureModel],
                 pooling_type: str = 'max',
                 embedding_size: Optional[int] = None) -> None:
        super().__init__()
        self.models = nn.ModuleList(models)

        self.nb_models = len(self.models)
        assert self.nb_models > 0

        self.pooling_type = pooling_type
        assert self.pooling_type in {'max', 'min', 'sum', 'mixture'}

        self.embedding_size = embedding_size

        self.mixture_weights = None
        if self.pooling_type == 'mixture':
            assert self.embedding_size is not None
            self.mixture_weights = nn.Linear(self.embedding_size, self.nb_models)

    def pool(self,
             rel: Tensor,
             arg1: Optional[Tensor],
             arg2: Optional[Tensor],
             xs: List[Tensor]) -> Tensor:
        res = None

        if self.pooling_type == 'max':
            for x in xs:
                res = x if res is None else torch.max(res, x)

        elif self.pooling_type == 'min':
            for x in xs:
                res = x if res is None else torch.min(res, x)

        elif self.pooling_type == 'sum':
            for x in xs:
                res = x if res is None else res + x

        elif self.pooling_type == 'mixture':
            # [B, M]
            attention = torch.softmax(self.mixture_weights(rel), dim=1)
            res_shape, batch_size = xs[0].shape, rel.shape[0]

            # [M, B, N]
            xs_tensor = torch.cat([x.view([1, batch_size, -1]) for x in xs], dim=0)
            res = torch.einsum('bm,mbn->bn', attention, xs_tensor).view(res_shape)

        assert res is not None
        return res

    def score(self,
              rel: Tensor, arg1: Tensor, arg2: Tensor) -> Tensor:
        score_lst = [model.score(rel, arg1, arg2) for model in self.models]
        res = self.pool(rel, arg1, arg2, score_lst)
        return res

    def forward(self,
                rel: Tensor,
                arg1: Optional[Tensor],
                arg2: Optional[Tensor]) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        score_lst = [model.forward(rel, arg1, arg2) for model in self.models]

        score_sp = self.pool(rel, arg1, arg2, [x for (x, _) in score_lst])
        score_po = self.pool(rel, arg1, arg2, [y for (_, y) in score_lst])

        return score_sp, score_po

    def factor(self,
               embedding_vector: Tensor) -> Tensor:
        return self.models[0].factor(embedding_vector)

    def extra_factors(self,
                      rel: Tensor,
                      arg1: Optional[Tensor],
                      arg2: Optional[Tensor]) -> List[Tensor]:
        extra_factors_lst = [model.extra_factors(rel, arg1, arg2) for model in self.models]
        return [j for i in extra_factors_lst for j in i]
