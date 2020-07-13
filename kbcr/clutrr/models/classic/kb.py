# -*- coding: utf-8 -*-

import numpy as np

import torch
from torch import nn, Tensor

from kbcr.kernels import BaseKernel
from kbcr.indexing import NPSearchIndex as Index

from typing import List, Tuple, Optional

import logging

logger = logging.getLogger(__name__)


class NeuralKB(nn.Module):
    def __init__(self,
                 kernel: BaseKernel,
                 k: int = 5,
                 device: Optional[torch.device] = None,
                 refresh_interval: Optional[int] = None):
        super().__init__()

        self.kernel = kernel
        self.k = k
        self.device = device
        self.refresh_interval = refresh_interval

        self.index_sp = Index()
        self.index_po = Index()
        self.index_spo = Index()

        self.index_count = 0

    def to_tnsr(self, x: np.ndarray) -> Tensor:
        res = torch.from_numpy(x)
        if self.device is not None:
            res = res.to(self.device)
        return res

    def neighbors(self,
                  rel: Tensor, arg1: Optional[Tensor], arg2: Optional[Tensor],
                  fact_rel: Tensor, fact_arg1: Tensor, fact_arg2: Tensor,
                  is_spo: bool = False) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:

        # print(fact_rel.shape, fact_arg1.shape, fact_arg2.shape)

        k = min(self.k, fact_rel.shape[0])
        res_sp, res_po, res_spo = None, None, None

        condition = self.refresh_interval is not None and self.index_count % self.refresh_interval == 0
        if self.refresh_interval is None or condition:
            fact_emb_sp = torch.cat([fact_rel, fact_arg1], dim=1)
            fact_emb_po = torch.cat([fact_rel, fact_arg2], dim=1)
            fact_emb_spo = torch.cat([fact_rel, fact_arg1, fact_arg2], dim=1)

            self.index_sp.build(fact_emb_sp.cpu().detach().numpy())
            self.index_po.build(fact_emb_po.cpu().detach().numpy())
            self.index_spo.build(fact_emb_spo.cpu().detach().numpy())

            # print(rel.shape, arg1.shape if arg1 is not None else None, arg2.shape if arg2 is not None else None)

        if is_spo is True:
            batch_emb = torch.cat([rel, arg1, arg2], dim=1)
            res_spo = self.index_spo.query(batch_emb.cpu().detach().numpy(), k=k)
        else:
            if arg1 is not None:
                batch_emb = torch.cat([rel, arg1], dim=1)
                res_sp = self.index_sp.query(batch_emb.cpu().detach().numpy(), k=k)

            if arg2 is not None:
                batch_emb = torch.cat([rel, arg2], dim=1)
                res_po = self.index_po.query(batch_emb.cpu().detach().numpy(), k=k)

        self.index_count += 1
        return res_sp, res_po, res_spo

    def __call__(self,
                 rel: Tensor,
                 arg1: Optional[Tensor],
                 arg2: Optional[Tensor],
                 facts: List[Tensor]) -> Tuple[Tensor, Optional[Tensor]]:
        assert (arg1 is not None and arg2 is not None) or ((arg1 is None) ^ (arg2 is None))

        embedding_size = rel.shape[1]

        if arg1 is not None and arg2 is not None:
            res = (self.score(rel, arg1, arg2, facts), None)
        else:
            res_sp, res_po = self.forward(rel, arg1, arg2, facts)
            res = res_sp if res_sp is not None else res_po
            scores, subs = res
            subs = subs.view(-1, embedding_size)
            scores = scores.view(subs.shape[0])
            res = scores, subs

        return res

    def score(self,
              rel: Tensor, arg1: Tensor, arg2: Tensor,
              facts: List[Tensor]) -> Tensor:

        batch_size = rel.shape[0]
        assert len(facts) == 3
        f_rel, f_arg1, f_arg2 = facts

        # [B, K]
        _, _, neigh_spo = self.neighbors(rel, arg1, arg2, f_rel, f_arg1, f_arg2, is_spo=True)

        neigh_spo_tensor = self.to_tnsr(neigh_spo)
        k = neigh_spo_tensor.shape[1]

        # [KB, 3E]
        fact_emb = torch.cat([f_rel, f_arg1, f_arg2], dim=1)
        # [B, 3E]
        batch_emb = torch.cat([rel, arg1, arg2], dim=1)

        fact_layer = nn.Embedding.from_pretrained(embeddings=fact_emb, freeze=False)
        # [B, K, 3E]
        f_emb = fact_layer(neigh_spo_tensor)

        # [B * K, 3E]
        embedding_size = f_emb.shape[2]
        f_emb_2d = f_emb.view(-1, embedding_size)
        b_emb_2d = batch_emb.view(batch_size, 1, embedding_size).repeat(1, k, 1).view(-1, embedding_size)

        # [B, K]
        scores = self.kernel(b_emb_2d, f_emb_2d).view(batch_size, -1)
        assert scores.shape[1] == k

        # [B]
        res, _ = torch.max(scores, dim=1)
        return res

    def forward(self,
                rel: Tensor, arg1: Optional[Tensor], arg2: Optional[Tensor],
                facts: List[Tensor]) -> Tuple[Optional[Tuple[Tensor, Tensor]], Optional[Tuple[Tensor, Tensor]]]:
        batch_size = rel.shape[0]
        embedding_size = rel.shape[1]

        assert len(facts) == 3
        fact_rel, fact_arg1, fact_arg2 = facts

        # [B, K], [B, K]
        neigh_sp, neigh_po, _ = self.neighbors(rel, arg1, arg2, fact_rel, fact_arg1, fact_arg2)

        f_emb = torch.cat(facts, dim=1)
        f_emb_layer = nn.Embedding.from_pretrained(embeddings=f_emb, freeze=False)

        res_sp = res_po = None

        if arg1 is not None:
            arg2_emb_layer = nn.Embedding.from_pretrained(embeddings=fact_arg2, freeze=False)

            # [B, k]
            neigh_sp_tensor = self.to_tnsr(neigh_sp)
            k = neigh_sp_tensor.shape[1]

            # [B * k, E]
            batch_rel = rel.view(batch_size, 1, embedding_size).repeat(1, k, 1).view(-1, embedding_size)
            batch_arg1 = arg1.view(batch_size, 1, embedding_size).repeat(1, k, 1).view(-1, embedding_size)
            batch_arg2 = arg2_emb_layer(neigh_sp_tensor.view(-1))

            # [B * k, 3E]
            batch_emb = torch.cat([batch_rel, batch_arg1, batch_arg2], dim=1)

            # [B * k, 3E]
            fact_emb = f_emb_layer(neigh_sp_tensor.view(-1))

            # [B, k]
            scores_sp = self.kernel(batch_emb, fact_emb).view(batch_size, k)
            subs_sp = batch_arg2.view(batch_size, k, embedding_size)

            assert scores_sp.shape[0] == subs_sp.shape[0]
            assert scores_sp.shape[1] == subs_sp.shape[1]

            res_sp = (scores_sp, subs_sp)

        if arg2 is not None:
            arg1_emb_layer = nn.Embedding.from_pretrained(embeddings=fact_arg1, freeze=False)

            # [B, k]
            neigh_po_tensor = self.to_tnsr(neigh_po)
            k = neigh_po_tensor.shape[1]

            # [B * k, E]
            batch_rel = rel.view(batch_size, 1, embedding_size).repeat(1, k, 1).view(-1, embedding_size)
            batch_arg1 = arg1_emb_layer(neigh_po_tensor.view(-1))
            batch_arg2 = arg2.view(batch_size, 1, embedding_size).repeat(1, k, 1).view(-1, embedding_size)

            # [B * k, 3E]
            batch_emb = torch.cat([batch_rel, batch_arg1, batch_arg2], dim=1)

            # [B * k, 3E]
            fact_emb = f_emb_layer(neigh_po_tensor.view(-1))

            # [B, k]
            scores_po = self.kernel(batch_emb, fact_emb).view(batch_size, k)
            subs_po = batch_arg2.view(batch_size, k, embedding_size)

            assert scores_po.shape[0] == subs_po.shape[0]
            assert scores_po.shape[1] == subs_po.shape[1]

            res_po = (scores_po, subs_po)

        return res_sp, res_po

    def factor(self,
               embedding_vector: Tensor) -> Tensor:
        return embedding_vector
