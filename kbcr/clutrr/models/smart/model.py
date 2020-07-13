# -*- coding: utf-8 -*-

import torch
from torch import nn, Tensor

from kbcr.clutrr.models.smart.kb import NeuralKB
from kbcr.reformulators import BaseReformulator

from kbcr.clutrr.models.smart.util import do_cmp as do

from typing import Tuple, Optional, List

import logging

logger = logging.getLogger(__name__)


class Hoppy(nn.Module):
    def __init__(self,
                 model: NeuralKB,
                 hops_lst: List[Tuple[BaseReformulator, bool]],
                 depth: int = 0):
        super().__init__()
        self.model: NeuralKB = model

        self.depth = depth
        assert self.depth >= 0

        self._hops_lst = nn.ModuleList([hops for hops, _ in hops_lst])
        self.hops_lst = hops_lst

    def _tnorm(self, x: Tensor, y: Tensor):
        return torch.min(x, y)

    def r_hop(self,
              rel: Tensor,
              arg1: Optional[Tensor],
              arg2: Optional[Tensor],
              facts: List[Tensor],
              depth: int) -> Tuple[Tensor, Tensor]:
        assert (arg1 is None) ^ (arg2 is None)
        assert depth >= 0

        batch_size, embedding_size = rel.shape[0], rel.shape[1]

        # [B, N]
        res_sp, res_po = self.r_forward(rel, arg1, arg2, facts, depth=depth)
        res = res_sp if arg2 is None else res_po

        # [B, K], [B, K, E]
        scores, subs = res

        assert scores.shape[0] == subs.shape[0], f'{scores.shape, subs.shape}'
        assert scores.shape[1] == subs.shape[1], f'{scores.shape, subs.shape}'

        assert scores.shape[0] == batch_size
        assert subs.shape[2] == embedding_size

        # [B, K], [B, K, E]
        return res

    def score(self,
              rel: Tensor,
              arg1: Tensor,
              arg2: Tensor,
              facts: List[Tensor]) -> Tensor:
        return self.r_score(rel, arg1, arg2, facts, depth=self.depth)

    def r_score(self,
                rel: Tensor,
                arg1: Tensor,
                arg2: Tensor,
                facts: List[Tensor],
                depth: int) -> Tensor:
        res = None
        for d in range(depth + 1):
            scores = self.depth_r_score(rel, arg1, arg2, facts, depth=d)
            res = scores if res is None else torch.max(res, scores)
        return res

    def depth_r_score(self,
                      rel: Tensor,
                      arg1: Tensor,
                      arg2: Tensor,
                      facts: List[Tensor],
                      depth: int) -> Tensor:
        assert depth >= 0

        if depth == 0:
            return self.model.score(rel, arg1, arg2, facts)

        batch_size, embedding_size = rel.shape[0], rel.shape[1]
        global_res = None

        for hops_generator, is_reversed in self.hops_lst:
            sources, scores = arg1, None
            hop_rel_lst = hops_generator(rel)
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
                    if is_reversed:
                        z_scores, z_emb = self.r_hop(hop_rel_2d, None, sources_2d, facts, depth=depth - 1)
                    else:
                        z_scores, z_emb = self.r_hop(hop_rel_2d, sources_2d, None, facts, depth=depth - 1)
                    k = z_emb.shape[1]

                    # [B * S * K]
                    z_scores_1d = z_scores.view(-1)
                    # [B * S * K, E]
                    z_emb_2d = z_emb.view(-1, embedding_size)

                    # [B * S * K, E]
                    sources = z_emb_2d
                    # [B * S * K]
                    scores = z_scores_1d if scores is None \
                        else self._tnorm(z_scores_1d, scores.view(-1, 1).repeat(1, k).view(-1))
                else:
                    # [B, S, E]
                    arg2_3d = arg2.view(-1, 1, embedding_size).repeat(1, nb_branches, 1)
                    # [B * S, E]
                    arg2_2d = arg2_3d.view(-1, embedding_size)

                    # [B * S]
                    if is_reversed:
                        z_scores_1d = self.r_score(hop_rel_2d, arg2_2d, sources_2d, facts, depth=depth - 1)
                    else:
                        z_scores_1d = self.r_score(hop_rel_2d, sources_2d, arg2_2d, facts, depth=depth - 1)

                    scores = z_scores_1d if scores is None else self._tnorm(z_scores_1d, scores)

            if scores is not None:
                scores_2d = scores.view(batch_size, -1)
                res, _ = torch.max(scores_2d, dim=1)
            else:
                res = self.model.score(rel, arg1, arg2, facts)

            global_res = res if global_res is None else torch.max(global_res, res)

        return global_res

    def forward(self,
                rel: Tensor,
                arg1: Optional[Tensor],
                arg2: Optional[Tensor],
                facts: List[Tensor]) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        res_sp, res_po = self.r_forward(rel, arg1, arg2, facts, depth=self.depth)
        return res_sp, res_po

    def r_forward(self,
                  rel: Tensor, arg1: Optional[Tensor], arg2: Optional[Tensor],
                  facts: List[Tensor],
                  depth: int) -> Tuple[Optional[Tuple[Tensor, Tensor]], Optional[Tuple[Tensor, Tensor]]]:
        res_sp, res_po = None, None
        for d in range(depth + 1):
            tmp_res_sp, tmp_res_po = self.depth_r_forward(rel, arg1, arg2, facts, depth=d)

            res_sp = tmp_res_sp if res_sp is None else do(res_sp, tmp_res_sp)
            res_po = tmp_res_po if res_po is None else do(res_po, tmp_res_po)

        return res_sp, res_po

    def depth_r_forward(self,
                        rel: Tensor,
                        arg1: Optional[Tensor],
                        arg2: Optional[Tensor],
                        facts: List[Tensor],
                        depth: int) -> Tuple[Optional[Tuple[Tensor, Tensor]], Optional[Tuple[Tensor, Tensor]]]:
        batch_size, embedding_size = rel.shape[0], rel.shape[1]

        if depth == 0:
            return self.model.forward(rel, arg1, arg2, facts)

        glob_res_sp = None
        glob_res_po = None

        for hop_generators, is_reversed in self.hops_lst:
            hop_rel_lst = hop_generators(rel)
            nb_hops = len(hop_rel_lst)

            if arg1 is not None:
                res_sp = None
                sources = arg1
                scores = None

                for hop_idx, hop_rel in enumerate(hop_rel_lst, start=1):
                    # [B * S, E]
                    sources_2d = sources.view(-1, embedding_size)
                    nb_sources = sources_2d.shape[0]

                    nb_branches = nb_sources // batch_size

                    hop_rel_3d = hop_rel.view(-1, 1, embedding_size).repeat(1, nb_branches, 1)
                    hop_rel_2d = hop_rel_3d.view(-1, embedding_size)

                    if hop_idx < nb_hops:
                        # [B * S, K], [B * S, K, E]
                        if is_reversed:
                            z_scores, z_emb = self.r_hop(hop_rel_2d, None, sources_2d, facts, depth=depth - 1)
                        else:
                            z_scores, z_emb = self.r_hop(hop_rel_2d, sources_2d, None, facts, depth=depth - 1)

                        k = z_emb.shape[1]

                        # [B * S * K]
                        z_scores_1d = z_scores.view(-1)
                        # [B * S * K, E]
                        z_emb_2d = z_emb.view(-1, embedding_size)

                        # [B * S * K, E]
                        sources = z_emb_2d
                        # [B * S * K]
                        if scores is None:
                            scores = z_scores_1d
                        else:
                            scores = self._tnorm(z_scores_1d, scores.view(-1, 1).repeat(1, k).view(-1))
                    else:
                        # [B * S, K]
                        if is_reversed:
                            _, res_sp = self.r_forward(hop_rel_2d, None, sources_2d, facts, depth=depth - 1)
                        else:
                            res_sp, _ = self.r_forward(hop_rel_2d, sources_2d, None, facts, depth=depth - 1)

                        # [B * S, K], [B * S, K, E]
                        scores_sp, emb_sp = res_sp

                        assert scores_sp.shape[0] == emb_sp.shape[0]
                        assert scores_sp.shape[1] == emb_sp.shape[1]

                        k = scores_sp.shape[1]
                        emb_size = emb_sp.shape[2]

                        if scores is not None:
                            scores = scores.view(-1, 1).repeat(1, k)
                            scores_sp = self._tnorm(scores, scores_sp)

                            # [B, S, K]
                            scores_sp = scores_sp.view(batch_size, -1, k)
                            # [B, K], [B, K]
                            scores_sp, ind_sp = torch.max(scores_sp, dim=1)

                            # [B, 1, K]
                            ind_4d_sp = ind_sp.view(batch_size, 1, k, 1)
                            emb_4d_sp = emb_sp.view(batch_size, -1, k, emb_size)
                            emb_sp = torch.gather(emb_4d_sp, 1, ind_4d_sp.repeat(1, 1, 1, emb_size))
                            emb_sp = emb_sp.view(batch_size, k, emb_size)

                        res_sp = (scores_sp, emb_sp)

                if glob_res_sp is None:
                    glob_res_sp = res_sp
                else:
                    glob_res_sp = do(glob_res_sp, res_sp)

            if arg2 is not None:
                res_po = None
                sources = arg2
                scores = None

                for hop_idx, hop_rel in enumerate(reversed([h for h in hop_rel_lst]), start=1):
                    # [B * S, E]
                    sources_2d = sources.view(-1, embedding_size)
                    nb_sources = sources_2d.shape[0]

                    nb_branches = nb_sources // batch_size

                    hop_rel_3d = hop_rel.view(-1, 1, embedding_size).repeat(1, nb_branches, 1)
                    hop_rel_2d = hop_rel_3d.view(-1, embedding_size)

                    if hop_idx < nb_hops:
                        # [B * S, K], [B * S, K, E]
                        if is_reversed:
                            z_scores, z_emb = self.r_hop(hop_rel_2d, sources_2d, None, facts, depth=depth - 1)
                        else:
                            z_scores, z_emb = self.r_hop(hop_rel_2d, None, sources_2d, facts, depth=depth - 1)
                        k = z_emb.shape[1]

                        # [B * S * K]
                        z_scores_1d = z_scores.view(-1)
                        # [B * S * K, E]
                        z_emb_2d = z_emb.view(-1, embedding_size)

                        # [B * S * K, E]
                        sources = z_emb_2d
                        # [B * S * K]
                        scores = z_scores_1d if scores is None \
                            else self._tnorm(z_scores_1d, scores.view(-1, 1).repeat(1, k).view(-1))
                    else:
                        # [B * S, N]
                        if is_reversed:
                            res_po, _ = self.r_forward(hop_rel_2d, sources_2d, None, facts, depth=depth - 1)
                        else:
                            _, res_po = self.r_forward(hop_rel_2d, None, sources_2d, facts, depth=depth - 1)

                        scores_po, emb_po = res_po

                        assert scores_po.shape[0] == emb_po.shape[0]
                        assert scores_po.shape[1] == emb_po.shape[1]

                        k = scores_po.shape[1]
                        emb_size = emb_po.shape[2]

                        if scores is not None:
                            scores = scores.view(-1, 1).repeat(1, k)
                            scores_po = self._tnorm(scores, scores_po)

                            # [B, S, K]
                            scores_po = scores_po.view(batch_size, -1, k)
                            # [B, K]
                            scores_po, ind_po = torch.max(scores_po, dim=1)

                            # [B, K, E]
                            ind_4d_po = ind_po.view(batch_size, 1, k, 1)
                            emb_4d_po = emb_po.view(batch_size, -1, k, emb_size)
                            emb_po = torch.gather(emb_4d_po, 1, ind_4d_po.repeat(1, 1, 1, emb_size))
                            emb_po = emb_po.view(batch_size, k, emb_size)

                        res_po = (scores_po, emb_po)

                if glob_res_po is None:
                    glob_res_po = res_po
                else:
                    glob_res_po = do(glob_res_po, res_po)

        if glob_res_sp is None and glob_res_po is None:
            glob_res_sp, glob_res_po = self.model.forward(rel, arg1, arg2)

        return glob_res_sp, glob_res_po

    def factor(self,
               embedding_vector: Tensor) -> Tensor:
        return self.model.factor(embedding_vector)

    def extra_factors(self,
                      rel: Tensor,
                      arg1: Optional[Tensor],
                      arg2: Optional[Tensor]) -> List[Tensor]:
        return [hop_generator(rel) for hop_generators in self.hops_lst for hop_generator in hop_generators]
