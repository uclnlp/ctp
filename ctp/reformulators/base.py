# -*- coding: utf-8 -*-

import torch
from torch import nn, Tensor

from abc import ABC, abstractmethod

from ctp.kernels import BaseKernel
from ctp.reformulators.util import AttentiveLinear

from typing import List, Optional

import logging

logger = logging.getLogger(__name__)


class BaseReformulator(ABC, nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, rel: Tensor) -> List[Tensor]:
        raise NotImplementedError

    @abstractmethod
    def forward(self, rel: Tensor) -> List[Tensor]:
        raise NotImplementedError

    def prior(self, rel: Tensor) -> Optional[Tensor]:
        # batch_size = rel.shape[0]
        # res = torch.ones(batch_size, dtype=rel.dtype)
        res = None
        return res


class StaticReformulator(BaseReformulator):
    def __init__(self,
                 nb_hops: int,
                 embedding_size: int,
                 init_name: Optional[str] = "uniform",
                 lower_bound: float = -1.0,
                 upper_bound: float = 1.0):
        super().__init__()

        self.nb_hops = nb_hops
        self.embedding_size = embedding_size
        self.init_name = init_name

        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        def make_memory() -> Tensor:
            return self._param(1, self.embedding_size)
        self.memory_lst = nn.ParameterList([make_memory() for _ in range(self.nb_hops)])

    def _param(self, dim_in: int, dim_out: int, std: float = 1.0) -> Tensor:
        w = torch.zeros(dim_in, dim_out)

        if self.init_name in {"uniform"}:
            nn.init.uniform_(w, self.lower_bound, self.upper_bound)
        elif self.init_name in {"normal"}:
            nn.init.normal_(w, std=std)

        return nn.Parameter(w, requires_grad=True)

    def forward(self, rel: Tensor) -> List[Tensor]:
        batch_size = rel.shape[0]
        res = [memory.repeat(batch_size, 1) for memory in self.memory_lst]
        return res


class NTPReformulator(StaticReformulator):
    def __init__(self,
                 kernel: BaseKernel,
                 **kwargs):
        super().__init__(**kwargs)
        self.kernel = kernel
        self.head = self._param(1, self.embedding_size)

    def forward(self, rel: Tensor) -> List[Tensor]:
        batch_size = rel.shape[0]
        res = [memory.repeat(batch_size, 1) for memory in self.memory_lst]
        return res

    def prior(self, rel: Tensor) -> Tensor:
        res = self.kernel(rel, self.head)
        return res.view(-1)


class GNTPReformulator(BaseReformulator):
    def __init__(self,
                 kernel: BaseKernel,
                 head: Tensor,
                 body: List[Tensor]):
        super().__init__()
        self.kernel = kernel
        self.head = head
        self.body = body

    def forward(self, rel: Tensor) -> List[Tensor]:
        for e in self.body:
            # print(rel.shape, e.shape)
            assert rel.shape[0] == e.shape[0]
        return self.body

    def prior(self, rel: Tensor) -> Tensor:
        assert rel.shape[0] == self.head.shape[0]
        res = self.kernel(rel, self.head)
        # print('XXX', rel.shape, self.head.shape, res.shape)
        return res.view(-1)


class LinearReformulator(BaseReformulator):
    def __init__(self,
                 nb_hops: int,
                 embedding_size: int,
                 init_name: Optional[str] = "uniform",
                 lower_bound: float = -1.0,
                 upper_bound: float = 1.0):
        super().__init__()

        self.nb_hops = nb_hops
        self.embedding_size = embedding_size
        self.init_name = init_name

        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        def make_hop(std: float = 1.0) -> nn.Module:
            res = nn.Linear(embedding_size, embedding_size)

            if self.init_name in {"uniform"}:
                nn.init.uniform_(res.weight, self.lower_bound, self.upper_bound)
            elif self.init_name in {"normal"}:
                nn.init.normal_(res.weight, std=std)

            return res

        self.hops_lst = nn.ModuleList([make_hop() for _ in range(nb_hops)])

    def forward(self, rel: Tensor) -> List[Tensor]:
        res = [hop(rel) for hop in self.hops_lst]
        return res


class AttentiveReformulator(BaseReformulator):
    def __init__(self,
                 nb_hops: int,
                 embeddings: nn.Embedding,
                 init_name: Optional[str] = "uniform",
                 lower_bound: float = -1.0,
                 upper_bound: float = 1.0):
        super().__init__()

        self.nb_hops = nb_hops
        self.embeddings = embeddings
        self.init_name = init_name

        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        def make_hop(std: float = 1.0) -> nn.Module:
            res = AttentiveLinear(embeddings)

            if self.init_name in {"uniform"}:
                nn.init.uniform_(res.projection.weight, self.lower_bound, self.upper_bound)
            elif self.init_name in {"normal"}:
                nn.init.normal_(res.projection.weight, std=std)

            return res
        self.hops_lst = nn.ModuleList([make_hop() for _ in range(nb_hops)])

    def forward(self, rel: Tensor) -> List[Tensor]:
        res = [hop(rel) for hop in self.hops_lst]
        return res


class MemoryReformulator(BaseReformulator):

    class Memory(nn.Module):
        def __init__(self,
                     nb_hops: int,
                     nb_rules: int,
                     embedding_size: int,
                     init_name: Optional[str] = "uniform",
                     lower_bound: float = -1.0,
                     upper_bound: float = 1.0):
            super().__init__()
            self.nb_hops = nb_hops
            self.nb_rules = nb_rules
            self.embedding_size = embedding_size
            self.init_name = init_name

            self.lower_bound = lower_bound
            self.upper_bound = upper_bound

            def _param(dim_in: int, dim_out: int, std: float = 1.0) -> Tensor:
                w = torch.zeros(dim_in, dim_out)

                if self.init_name in {"uniform"}:
                    nn.init.uniform_(w, self.lower_bound, self.upper_bound)
                elif self.init_name in {"normal"}:
                    nn.init.normal_(w, std=std)

                return nn.Parameter(w, requires_grad=True)

            def make_memory() -> Tensor:
                return _param(self.nb_rules, self.embedding_size)

            self.memory = nn.ParameterList([make_memory() for _ in range(self.nb_hops)])

    def __init__(self,
                 memory: Memory):
        # Note: this init_name governs the attention weights over the memory, I think it's fine to keep it as it is
        super().__init__()
        self.memory = memory
        self.projection = nn.Linear(self.memory.embedding_size, self.memory.nb_rules)

    def forward(self, rel: Tensor) -> List[Tensor]:
        attn_logits = self.projection(rel)
        attn = torch.softmax(attn_logits, dim=1)
        res = [attn @ cell for cell in self.memory.memory]
        return res


class SymbolicReformulator(BaseReformulator):
    def __init__(self,
                 embeddings: nn.Embedding,
                 indices: Tensor):
        super().__init__()
        self.embeddings = embeddings
        self.indices = indices

    def forward(self, rel: Tensor) -> List[Tensor]:
        batch_size = rel.shape[0]
        # [I, E]
        res = self.embeddings(self.indices)
        nb_indices, embedding_size = res.shape[0], res.shape[1]
        # [B, I, E]
        res = res.view(1, nb_indices, embedding_size).repeat(batch_size, 1, 1)

        # res = list(torch.split(res, split_size_or_sections=1, dim=1))
        res = [a.view(res.shape[0], res.shape[2]) for a in torch.split(res, split_size_or_sections=1, dim=1)]
        return res
