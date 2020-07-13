# -*- coding: utf-8 -*-

import numpy as np

import torch
from torch import nn, Tensor

from kbcr.kernels import BaseKernel
from kbcr.smart.base import BaseSmartModel

from kbcr.indexing import NPSearchIndex
from kbcr.indexing import FAISSSearchIndex
from kbcr.indexing import NMSSearchIndex

from typing import List, Tuple, Optional

import logging

logger = logging.getLogger(__name__)


class NeuralKB(BaseSmartModel):
    def __init__(self,
                 entity_embeddings: nn.Embedding,
                 predicate_embeddings: nn.Embedding,
                 facts: List[Tensor],
                 kernel: BaseKernel,
                 k: int = 5,
                 device: Optional[torch.device] = None,
                 index_type: str = 'np',
                 refresh_interval: Optional[int] = None) -> None:
        super().__init__()

        self.entity_embeddings = entity_embeddings
        self.predicate_embeddings = predicate_embeddings

        self.facts = facts
        assert len(self.facts) == 3

        self.kernel = kernel
        self.k = k

        self.device = device

        self.index_type = index_type
        self.refresh_interval = refresh_interval

        index_factory = {
            'faiss': lambda: FAISSSearchIndex(),
            'np': lambda: NPSearchIndex(),
            'nms': lambda: NMSSearchIndex()
        }

        assert self.index_type in index_factory

        self.index_sp = index_factory[self.index_type]()
        self.index_po = index_factory[self.index_type]()
        self.index_spo = index_factory[self.index_type]()

        self.index_count = 0

    def to_tnsr(self, x: np.ndarray) -> Tensor:
        res = torch.from_numpy(x).type(torch.long)
        if self.device is not None:
            res = res.to(self.device)
        return res

    def generate_facts(self) -> Tuple[Tensor, Tensor, Tensor]:
        fact_rel, fact_arg1, fact_arg2 = self.facts
        emb_rel = self.predicate_embeddings(fact_rel)
        emb_arg1 = self.entity_embeddings(fact_arg1)
        emb_arg2 = self.entity_embeddings(fact_arg2)
        return emb_rel, emb_arg1, emb_arg2

    def score(self,
              rel: Tensor, arg1: Tensor, arg2: Tensor,
              mask_indices: Optional[Tensor] = None,
              *args, **kwargs) -> Tensor:
        f_rel_emb, f_arg1_emb, f_arg2_emb = self.generate_facts()

        batch_size = rel.shape[0]

        # [B, K]
        _, _, neigh_spo = self.neighbors(rel, arg1, arg2, f_rel_emb, f_arg1_emb, f_arg2_emb, is_spo=True)

        neigh_spo_tensor = self.to_tnsr(neigh_spo)
        k = neigh_spo_tensor.shape[1]

        # [KB, 3E]
        fact_emb = torch.cat([f_rel_emb, f_arg1_emb, f_arg2_emb], dim=1)
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

        mask_ind = None
        if mask_indices is not None:
            repeat_mask = batch_size // mask_indices.shape[0]
            mask_ind = mask_indices.view(-1, 1).repeat(1, repeat_mask).view(-1)

        if mask_ind is not None:
            mask = 1.0 - (mask_ind.view(-1, 1).repeat(1, k) == neigh_spo_tensor).float()
            scores = scores * mask

        # [B]
        res, _ = torch.max(scores, dim=1)
        return res

    def neighbors(self,
                  rel: Tensor, arg1: Optional[Tensor], arg2: Optional[Tensor],
                  fact_rel: Tensor, fact_arg1: Tensor, fact_arg2: Tensor,
                  is_spo: bool = False) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        k = min(self.k, fact_rel.shape[0])
        res_sp, res_po, res_spo = None, None, None

        condition = self.refresh_interval is not None and self.index_count % self.refresh_interval == 0
        if self.refresh_interval is None or condition:
            fact_emb_sp = torch.cat([fact_rel, fact_arg1], dim=1)
            fact_emb_po = torch.cat([fact_rel, fact_arg2], dim=1)
            fact_emb_spo = torch.cat([fact_rel, fact_arg1, fact_arg2], dim=1)

            # logger.info('Refreshing indexes ..')

            self.index_sp.build(fact_emb_sp.cpu().detach().numpy())
            self.index_po.build(fact_emb_po.cpu().detach().numpy())
            self.index_spo.build(fact_emb_spo.cpu().detach().numpy())

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

    def forward(self,
                rel: Tensor,
                arg1: Optional[Tensor],
                arg2: Optional[Tensor],
                mask_indices: Optional[Tensor] = None,
                *args, **kwargs) -> Tuple[Optional[Tuple[Tensor, Tensor]], Optional[Tuple[Tensor, Tensor]]]:
        batch_size = rel.shape[0]
        embedding_size = rel.shape[1]

        # [KB, E], [KB, E], [KB, E]
        f_rel_emb, f_arg1_emb, f_arg2_emb = self.generate_facts()

        mask_ind = None
        if mask_indices is not None:
            repeat_mask = batch_size // mask_indices.shape[0]
            mask_ind = mask_indices.view(-1, 1).repeat(1, repeat_mask).view(-1)

        # [B, K], [B, K]
        neigh_sp, neigh_po, _ = self.neighbors(rel, arg1, arg2, f_rel_emb, f_arg1_emb, f_arg2_emb)

        # [KB, 3E]
        f_emb = torch.cat([f_rel_emb, f_arg1_emb, f_arg2_emb], dim=1)
        f_emb_layer = nn.Embedding.from_pretrained(embeddings=f_emb, freeze=False)

        res_sp = res_po = None

        if arg1 is not None:
            arg2_emb_layer = nn.Embedding.from_pretrained(embeddings=f_arg2_emb, freeze=False)

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

            if mask_ind is not None:
                mask = 1.0 - (mask_ind.view(-1, 1).repeat(1, k) == neigh_sp_tensor).float()
                scores_sp = scores_sp * mask

            assert scores_sp.shape[0] == subs_sp.shape[0]
            assert scores_sp.shape[1] == subs_sp.shape[1]

            res_sp = (scores_sp, subs_sp)

        if arg2 is not None:
            arg1_emb_layer = nn.Embedding.from_pretrained(embeddings=f_arg1_emb, freeze=False)

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

            if mask_ind is not None:
                mask = 1.0 - (mask_ind.view(-1, 1).repeat(1, k) == neigh_po_tensor).float()
                scores_po = scores_po * mask

            assert scores_po.shape[0] == subs_po.shape[0]
            assert scores_po.shape[1] == subs_po.shape[1]

            res_po = (scores_po, subs_po)

        return res_sp, res_po

    def forward_(self,
                 rel: Tensor,
                 arg1: Optional[Tensor],
                 arg2: Optional[Tensor],
                 *args, **kwargs) -> Tuple[Optional[Tuple[Tensor, Optional[Tensor]]],
                                           Optional[Tuple[Tensor, Optional[Tensor]]]]:
        # [N, E]
        emb = self.entity_embeddings.weight

        batch_size = rel.shape[0]
        nb_entities = emb.shape[0]

        fact_emb_lst = self.generate_facts()

        score_sp_lst, score_po_lst = [], []

        for i in range(batch_size):
            i_emb_rel = rel[i, :].view(1, -1).repeat(nb_entities, 1)

            if arg1 is not None:
                # [N, E]
                i_emb_arg1 = arg1[i, :].view(1, -1).repeat(nb_entities, 1)

                # [KB, 3E]
                fact_emb = torch.cat(fact_emb_lst, dim=1)
                # [N, 3E]
                batch_emb = torch.cat([i_emb_rel, i_emb_arg1, emb], dim=1)

                # [B, KB]
                pairwise_sp = self.kernel.pairwise(batch_emb, fact_emb)

                i_score_sp, _ = torch.max(pairwise_sp, dim=1)
                score_sp_lst += [i_score_sp.view(1, -1)]

            if arg2 is not None:
                i_emb_arg2 = arg2[i, :].view(1, -1).repeat(nb_entities, 1)

                # [KB, 3E]
                fact_emb = torch.cat(fact_emb_lst, dim=1)
                # [N, 3E]
                batch_emb = torch.cat([i_emb_rel, emb, i_emb_arg2], dim=1)

                # [B, KB]
                pairwise_po = self.kernel.pairwise(batch_emb, fact_emb)

                i_score_po, _ = torch.max(pairwise_po, dim=1)
                score_po_lst += [i_score_po.view(1, -1)]

        # [B, N]
        score_sp = torch.cat(score_sp_lst, dim=0) if arg1 is not None else None
        score_po = torch.cat(score_po_lst, dim=0) if arg2 is not None else None

        return (score_sp, None), (score_po, None)

    def forward__(self,
                  rel: Tensor,
                  arg1: Optional[Tensor],
                  arg2: Optional[Tensor],
                  *args, **kwargs) -> Tuple[Optional[Tuple[Tensor, Optional[Tensor]]],
                                            Optional[Tuple[Tensor, Optional[Tensor]]]]:
        batch_size = rel.shape[0]
        emb = self.entity_embeddings.weight

        nb_entities = emb.shape[0]

        fact_emb_lst = self.generate_facts()
        f_rel_emb, f_arg1_emb, f_arg2_emb = fact_emb_lst

        f_emb = torch.cat(fact_emb_lst, dim=1)
        f_emb_layer = nn.Embedding.from_pretrained(embeddings=f_emb, freeze=False)

        score_sp_lst, score_po_lst = [], []

        for i in range(batch_size):
            i_emb_rel = rel[i, :].view(1, -1).repeat(nb_entities, 1)

            if arg1 is not None:
                # [N, E]
                i_emb_arg1 = arg1[i, :].view(1, -1).repeat(nb_entities, 1)

                # Embedding of (S, P, all entities)
                # [N, 3E]
                batch_emb = torch.cat([i_emb_rel, i_emb_arg1, emb], dim=1)

                # For all corruptions, we have the closest K facts
                # [N, K]
                _, _, neigh_spo = self.neighbors(i_emb_rel, i_emb_arg1, emb,
                                                 f_rel_emb, f_arg1_emb, f_arg2_emb,
                                                 is_spo=True)
                neigh_spo_tensor = self.to_tnsr(neigh_spo)
                k = neigh_spo_tensor.shape[1]

                # First look up these facts
                # [N, K, 3E]
                tmp = f_emb_layer(neigh_spo_tensor.view(-1)).view(nb_entities, k, -1)

                # Then repeat (S, P, all entities) K times
                batch_emb = batch_emb.view(nb_entities, 1, -1).repeat(1, k, 1)

                # [N, K]
                pairwise_sp = self.kernel(batch_emb, tmp)
                pairwise_sp = pairwise_sp.view(batch_emb.shape[0], batch_emb.shape[1])

                i_score_sp, _ = torch.max(pairwise_sp, dim=1)
                score_sp_lst += [i_score_sp.view(1, -1)]

            if arg2 is not None:
                i_emb_arg2 = arg2[i, :].view(1, -1).repeat(nb_entities, 1)

                # [N, 3E]
                batch_emb = torch.cat([i_emb_rel, emb, i_emb_arg2], dim=1)

                # For all corruptions, we have the closest K facts
                # [N, K]
                _, _, neigh_spo = self.neighbors(i_emb_rel, emb, i_emb_arg2,
                                                 f_rel_emb, f_arg1_emb, f_arg2_emb,
                                                 is_spo=True)
                neigh_spo_tensor = self.to_tnsr(neigh_spo)
                k = neigh_spo_tensor.shape[1]

                # First look up these facts
                # [N, K, 3E]
                tmp = f_emb_layer(neigh_spo_tensor.view(-1)).view(nb_entities, k, -1)

                # Then repeat (S, P, all entities) K times
                batch_emb = batch_emb.view(nb_entities, 1, -1).repeat(1, k, 1)

                # [N, K]
                pairwise_po = self.kernel(batch_emb, tmp)
                pairwise_po = pairwise_po.view(batch_emb.shape[0], batch_emb.shape[1])

                i_score_po, _ = torch.max(pairwise_po, dim=1)
                score_po_lst += [i_score_po.view(1, -1)]

        # [B, N]
        score_sp = torch.cat(score_sp_lst, dim=0) if arg1 is not None else None
        score_po = torch.cat(score_po_lst, dim=0) if arg2 is not None else None

        return (score_sp, None), (score_po, None)

    def factor(self,
               embedding_vector: Tensor) -> Tensor:
        return embedding_vector
