# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod

import torch
from torch import nn, Tensor

from typing import List

import logging

logger = logging.getLogger(__name__)


class Regularizer(nn.Module, ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def __call__(self,
                 factors: List[Tensor]):
        raise NotImplementedError


class N2(Regularizer):
    def __init__(self):
        super().__init__()

    def __call__(self,
                 factors: List[Tensor]):
        norm = sum(torch.sum(torch.norm(f, 2, 1) ** 3) for f in factors)
        return norm / factors[0].shape[0]


class N3(Regularizer):
    def __init__(self):
        super().__init__()

    def __call__(self,
                 factors: List[Tensor]):
        norm = sum(torch.sum(torch.abs(f) ** 3) for f in factors)
        return norm / factors[0].shape[0]


class Entropy(Regularizer):
    def __init__(self, use_logits: bool = False) -> None:
        super().__init__()
        self.use_logits = use_logits

    def __call__(self,
                 factors: List[Tensor]):
        if self.use_logits is True:
            # Inputs are logits - turn them into probabilities
            factors = [torch.softmax(f, dim=1) for f in factors]
        res = sum(torch.sum(- torch.log(f) * f) for f in factors)
        return res / factors[0].shape[0]
