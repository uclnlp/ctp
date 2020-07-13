# -*- coding: utf-8 -*-

from torch import nn, Tensor

import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class BaseKernel(ABC, nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def __call__(self,
                 x: Tensor,
                 y: Tensor) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def pairwise(self,
                 x: Tensor,
                 y: Tensor) -> Tensor:
        raise NotImplementedError
