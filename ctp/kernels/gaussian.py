# -*- coding: utf-8 -*-

import torch
from torch import nn, Tensor

from ctp.kernels.base import BaseKernel

from typing import Optional, Tuple

import logging

logger = logging.getLogger(__name__)


class GaussianKernel(BaseKernel):
    def __init__(self,
                 slope: Optional[float] = 1.0,
                 boundaries: Optional[Tuple[float, float]] = None):
        super().__init__()

        self.slope = slope

        if self.slope is None:
            self.slope = nn.Parameter(torch.ones(1))

        self.boundaries = boundaries

    def __call__(self,
                 x: Tensor,
                 y: Tensor) -> Tensor:
        """
        :param x: [AxE] Tensor
        :param y: [AxE] Tensor
        :return: [A] Tensor
        """
        emb_size_x = x.shape[-1]
        emb_size_y = y.shape[-1]

        a = torch.reshape(x, [-1, emb_size_x])
        b = torch.reshape(y, [-1, emb_size_y])

        l2 = torch.sum((a - b) ** 2, dim=1)
        l2 = torch.clamp(l2, 1e-6, 1000)
        l2 = torch.sqrt(l2)

        res = torch.exp(- l2 * self.slope)

        if self.boundaries is not None:
            vmin, vmax = self.boundaries
            scaling_factor = vmax - vmin
            res = (res * scaling_factor) + vmin

        return res

    def pairwise(self,
                 x: Tensor,
                 y: Tensor) -> Tensor:
        """
        :param x: [AxE] Tensor
        :param y: [BxE] Tensor
        :return: [AxB] Tensor
        """
        dim_x, emb_size_x = x.shape[:-1], x.shape[-1]
        dim_y, emb_size_y = y.shape[:-1], y.shape[-1]

        a = torch.reshape(x, [-1, emb_size_x])
        b = torch.reshape(y, [-1, emb_size_y])

        c = - 2 * a @ b.t()
        na = torch.sum(a ** 2, dim=1, keepdim=True)
        nb = torch.sum(b ** 2, dim=1, keepdim=True)

        l2 = (c + nb.t()) + na
        l2 = torch.clamp(l2, 1e-6, 1000)
        l2 = torch.sqrt(l2)

        sim = torch.exp(- l2 * self.slope)
        res = torch.reshape(sim, dim_x + dim_y)

        if self.boundaries is not None:
            vmin, vmax = self.boundaries
            scaling_factor = vmax - vmin
            res = (res * scaling_factor) + vmin

        return res
