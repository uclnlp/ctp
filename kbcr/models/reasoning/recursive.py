# -*- coding: utf-8 -*-

import torch
from torch import nn, Tensor

from kbcr.models import BaseLatentFeatureModel
from kbcr.reformulators import BaseReformulator

from typing import Tuple, Optional, List

import logging

logger = logging.getLogger(__name__)


class RecursiveHoppy(BaseLatentFeatureModel):
    def __init__(self,
                 model: BaseLatentFeatureModel,
                 entity_embeddings: nn.Embedding,
                 hops: BaseReformulator,
                 is_reversed: bool = False,
                 k: int = 10,
                 depth: int = 0):
        super().__init__()

        self.model = model
        self.entity_embeddings = entity_embeddings
        self.hops = hops
        self.is_reversed = is_reversed
        self.k = k
        self.depth = depth

    def r_hop(self,
              rel: Tensor,
              arg1: Optional[Tensor],
              arg2: Optional[Tensor],
              depth: int) -> Tuple[Tensor, Tensor]:
        assert (arg1 is None) ^ (arg2 is None)
        assert depth >= 0

        batch_size, embedding_size = rel.shape[0], rel.shape[1]

        # [B, N]
        scores_sp, scores_po = self.r_forward(rel, arg1, arg2, depth=depth)
        scores = scores_sp if arg2 is None else scores_po

        k = min(self.k, scores.shape[1])

        # [B, K], [B, K]
        z_scores, z_indices = torch.topk(scores, k=k, dim=1)
        # [B, K, E]
        z_emb = self.entity_embeddings(z_indices)

        assert z_emb.shape[0] == batch_size
        assert z_emb.shape[2] == embedding_size

        return z_scores, z_emb

    def score(self,
              rel: Tensor,
              arg1: Tensor,
              arg2: Tensor,
              *args, **kwargs) -> Tensor:
        return self.r_score(rel, arg1, arg2, depth=self.depth)

    def r_score(self,
                rel: Tensor,
                arg1: Tensor,
                arg2: Tensor,
                depth: int) -> Tensor:
        res = None
        for d in range(depth + 1):
            scores = self.depth_r_score(rel, arg1, arg2, depth=d)
            res = scores if res is None else torch.max(res, scores)
        return res

    def depth_r_score(self,
                      rel: Tensor,
                      arg1: Tensor,
                      arg2: Tensor,
                      depth: int) -> Tensor:
        assert depth >= 0

        if depth == 0:
            return self.model.score(rel, arg1, arg2)

        batch_size, embedding_size = rel.shape[0], rel.shape[1]
        sources, scores = arg1, None

        hop_rel_lst = self.hops(rel)
        nb_hops = len(hop_rel_lst)

        for hop_idx, hop_rel in enumerate(hop_rel_lst, start=1):
            # [B * S, E]
            sources_2d = sources.view(-1, embedding_size)
            nb_sources = sources_2d.shape[0]

            nb_branches = nb_sources // batch_size

            hop_rel_3d = hop_rel.view(-1, 1, embedding_size).repeat(1, nb_branches, 1)
            hop_rel_2d = hop_rel_3d.view(-1, embedding_size)

            if hop_idx < nb_hops:
                # [B * S, K], [B * S, K, E]
                if self.is_reversed:
                    z_scores, z_emb = self.r_hop(hop_rel_2d, None, sources_2d, depth=depth - 1)
                else:
                    z_scores, z_emb = self.r_hop(hop_rel_2d, sources_2d, None, depth=depth - 1)

                k = z_emb.shape[1]

                # [B * S * K]
                z_scores_1d = z_scores.view(-1)
                # [B * S * K, E]
                z_emb_2d = z_emb.view(-1, embedding_size)

                # [B * S * K, E]
                sources = z_emb_2d
                # [B * S * K]
                scores = z_scores_1d if scores is None \
                    else torch.min(z_scores_1d, scores.view(-1, 1).repeat(1, k).view(-1))
            else:
                # [B, S, E]
                arg2_3d = arg2.view(-1, 1, embedding_size).repeat(1, nb_branches, 1)
                # [B * S, E]
                arg2_2d = arg2_3d.view(-1, embedding_size)

                # [B * S]
                if self.is_reversed:
                    z_scores_1d = self.r_score(hop_rel_2d, arg2_2d, sources_2d, depth=depth - 1)
                else:
                    z_scores_1d = self.r_score(hop_rel_2d, sources_2d, arg2_2d, depth=depth - 1)

                scores = z_scores_1d if scores is None else torch.min(z_scores_1d, scores)

        if scores is not None:
            scores_2d = scores.view(batch_size, -1)
            res, _ = torch.max(scores_2d, dim=1)
        else:
            res = self.model.score(rel, arg1, arg2)

        return res

    def forward(self,
                rel: Tensor,
                arg1: Optional[Tensor],
                arg2: Optional[Tensor],
                *args, **kwargs) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        return self.r_forward(rel, arg1, arg2, depth=self.depth)

    def r_forward(self,
                  rel: Tensor,
                  arg1: Optional[Tensor],
                  arg2: Optional[Tensor],
                  depth: int) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        res_sp, res_po = None, None
        for d in range(depth + 1):
            scores_sp, scores_po = self.depth_r_forward(rel, arg1, arg2, depth=d)
            res_sp = scores_sp if res_sp is None else torch.max(res_sp, scores_sp)
            res_po = scores_po if res_po is None else torch.max(res_po, scores_po)
        return res_sp, res_po

    def depth_r_forward(self,
                        rel: Tensor,
                        arg1: Optional[Tensor],
                        arg2: Optional[Tensor],
                        depth: int) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        batch_size, embedding_size = rel.shape[0], rel.shape[1]
        scores_sp = scores_po = None

        if depth == 0:
            return self.model.forward(rel, arg1, arg2)

        hop_rel_lst = self.hops(rel)
        nb_hops = len(hop_rel_lst)

        if arg1 is not None:
            sources, scores = arg1, None

            for hop_idx, hop_rel in enumerate(hop_rel_lst, start=1):
                # [B * S, E]
                sources_2d = sources.view(-1, embedding_size)
                nb_sources = sources_2d.shape[0]

                nb_branches = nb_sources // batch_size

                hop_rel_3d = hop_rel.view(-1, 1, embedding_size).repeat(1, nb_branches, 1)
                hop_rel_2d = hop_rel_3d.view(-1, embedding_size)

                if hop_idx < nb_hops:
                    # [B * S, K], [B * S, K, E]
                    if self.is_reversed:
                        z_scores, z_emb = self.r_hop(hop_rel_2d, None, sources_2d, depth=depth - 1)
                    else:
                        z_scores, z_emb = self.r_hop(hop_rel_2d, sources_2d, None, depth=depth - 1)

                    k = z_emb.shape[1]

                    # [B * S * K]
                    z_scores_1d = z_scores.view(-1)
                    # [B * S * K, E]
                    z_emb_2d = z_emb.view(-1, embedding_size)

                    # [B * S * K, E]
                    sources = z_emb_2d
                    # [B * S * K]
                    scores = z_scores_1d if scores is None \
                        else torch.min(z_scores_1d, scores.view(-1, 1).repeat(1, k).view(-1))
                else:
                    # [B * S, N]
                    if self.is_reversed:
                        _, scores_sp = self.r_forward(hop_rel_2d, None, sources_2d, depth=depth - 1)
                    else:
                        scores_sp, _ = self.r_forward(hop_rel_2d, sources_2d, None, depth=depth - 1)

                    nb_entities = scores_sp.shape[1]

                    if scores is not None:
                        scores = scores.view(-1, 1).repeat(1, nb_entities)
                        scores_sp = torch.min(scores, scores_sp)

                        # [B, S, N]
                        scores_sp = scores_sp.view(batch_size, -1, nb_entities)
                        # [B, N]
                        scores_sp, _ = torch.max(scores_sp, dim=1)

        if arg2 is not None:
            sources, scores = arg2, None

            for hop_idx, hop_rel in enumerate(reversed([h for h in hop_rel_lst]), start=1):
                # [B * S, E]
                sources_2d = sources.view(-1, embedding_size)
                nb_sources = sources_2d.shape[0]

                nb_branches = nb_sources // batch_size

                hop_rel_3d = hop_rel.view(-1, 1, embedding_size).repeat(1, nb_branches, 1)
                hop_rel_2d = hop_rel_3d.view(-1, embedding_size)

                if hop_idx < nb_hops:
                    # [B * S, K], [B * S, K, E]
                    if self.is_reversed:
                        z_scores, z_emb = self.r_hop(hop_rel_2d, sources_2d, None, depth=depth - 1)
                    else:
                        z_scores, z_emb = self.r_hop(hop_rel_2d, None, sources_2d, depth=depth - 1)

                    k = z_emb.shape[1]

                    # [B * S * K]
                    z_scores_1d = z_scores.view(-1)
                    # [B * S * K, E]
                    z_emb_2d = z_emb.view(-1, embedding_size)

                    # [B * S * K, E]
                    sources = z_emb_2d
                    # [B * S * K]
                    scores = z_scores_1d if scores is None \
                        else torch.min(z_scores_1d, scores.view(-1, 1).repeat(1, k).view(-1))
                else:
                    # [B * S, N]
                    if self.is_reversed:
                        scores_po, _ = self.r_forward(hop_rel_2d, sources_2d, None, depth=depth - 1)
                    else:
                        _, scores_po = self.r_forward(hop_rel_2d, None, sources_2d, depth=depth - 1)

                    nb_entities = scores_po.shape[1]

                    if scores is not None:
                        scores = scores.view(-1, 1).repeat(1, nb_entities)
                        scores_po = torch.min(scores, scores_po)

                        # [B, S, N]
                        scores_po = scores_po.view(batch_size, -1, nb_entities)
                        # [B, N]
                        scores_po, _ = torch.max(scores_po, dim=1)

        if scores_sp is None and scores_po is None:
            scores_sp, scores_po = self.model.forward(rel, arg1, arg2)

        return scores_sp, scores_po

    def factor(self,
               embedding_vector: Tensor) -> Tensor:
        return self.model.factor(embedding_vector)

    def extra_factors(self,
                      rel: Tensor,
                      arg1: Optional[Tensor],
                      arg2: Optional[Tensor]) -> List[Tensor]:
        return [hop for hop in self.hops(rel)]
