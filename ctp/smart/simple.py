# -*- coding: utf-8 -*-

import torch
from torch import nn, Tensor

from ctp.smart import BaseSmartModel
from ctp.reformulators import BaseReformulator

from typing import Tuple, Optional, List

import logging

logger = logging.getLogger(__name__)


class SimpleHoppy(BaseSmartModel):
    def __init__(self,
                 model: BaseSmartModel,
                 entity_embeddings: nn.Embedding,
                 hops_lst: List[Tuple[BaseReformulator, bool]]):
        super().__init__()

        self.model = model
        self.entity_embeddings = entity_embeddings
        self.hops_lst = hops_lst

        self._hops_lst = nn.ModuleList([hops for hops, _ in hops_lst])

    def hop(self,
            rel: Tensor,
            arg1: Optional[Tensor],
            arg2: Optional[Tensor],
            mask_indices: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        assert (arg1 is None) ^ (arg2 is None)
        batch_size, embedding_size = rel.shape[0], rel.shape[1]

        res_sp, res_po = self.model.forward(rel, arg1, arg2, mask_indices=mask_indices)
        res = res_sp if arg2 is None else res_po

        # [B, K], [B, K, E]
        scores, subs = res

        assert scores.shape[0] == subs.shape[0]
        assert scores.shape[1] == subs.shape[1]

        assert scores.shape[0] == batch_size
        assert subs.shape[2] == embedding_size

        # [B, K], [B, K, E]
        return res

    def score(self,
              rel: Tensor,
              arg1: Tensor,
              arg2: Tensor,
              mask_indices: Optional[Tensor] = None,
              *args, **kwargs) -> Tensor:
        batch_size, embedding_size = rel.shape[0], rel.shape[1]

        global_res = None

        for hops, is_reversed in self.hops_lst:
            sources, scores = arg1, None

            # XXX
            prior = hops.prior(rel)
            if prior is not None:
                scores = prior
            # scores = hops.prior(rel)

            hop_rel_lst = hops(rel)
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
                        z_scores, z_emb = self.hop(hop_rel_2d, None, sources_2d, mask_indices=mask_indices)
                    else:
                        z_scores, z_emb = self.hop(hop_rel_2d, sources_2d, None, mask_indices=mask_indices)

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
                        scores = torch.min(z_scores_1d, scores.view(-1, 1).repeat(1, k).view(-1))

                else:
                    # [B, S, E]
                    arg2_3d = arg2.view(-1, 1, embedding_size).repeat(1, nb_branches, 1)
                    # [B * S, E]
                    arg2_2d = arg2_3d.view(-1, embedding_size)

                    # [B * S]
                    if is_reversed:
                        z_scores_1d = self.model.score(hop_rel_2d, arg2_2d, sources_2d, mask_indices=mask_indices)
                    else:
                        z_scores_1d = self.model.score(hop_rel_2d, sources_2d, arg2_2d, mask_indices=mask_indices)

                    scores = z_scores_1d if scores is None else torch.min(z_scores_1d, scores)

            if scores is not None:
                scores_2d = scores.view(batch_size, -1)
                res, _ = torch.max(scores_2d, dim=1)
            else:
                res = self.model.score(rel, arg1, arg2, mask_indices=mask_indices)

            global_res = res if global_res is None else torch.max(global_res, res)

        return global_res

    def forward(self,
                rel: Tensor,
                arg1: Optional[Tensor],
                arg2: Optional[Tensor],
                mask_indices: Optional[Tensor] = None,
                *args, **kwargs) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        batch_size, embedding_size = rel.shape[0], rel.shape[1]
        scores_sp = scores_po = None

        global_scores_sp = global_scores_po = None

        for hops, is_reversed in self.hops_lst:
            hop_rel_lst = hops(rel)
            nb_hops = len(hop_rel_lst)

            if arg1 is not None:
                sources, scores = arg1, None

                # XXX
                prior = hops.prior(rel)
                if prior is not None:
                    scores = prior
                # scores = hops.prior(rel)

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
                            z_scores, z_emb = self.hop(hop_rel_2d, None, sources_2d, mask_indices=mask_indices)
                        else:
                            z_scores, z_emb = self.hop(hop_rel_2d, sources_2d, None, mask_indices=mask_indices)

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
                            scores = torch.min(z_scores_1d, scores.view(-1, 1).repeat(1, k).view(-1))
                    else:
                        # [B * S, K]
                        if is_reversed:
                            _, res_sp = self.model.forward(hop_rel_2d, None, sources_2d, mask_indices=mask_indices)
                        else:
                            res_sp, _ = self.model.forward(hop_rel_2d, sources_2d, None, mask_indices=mask_indices)

                        scores_sp, subs_sp = res_sp
                        k = scores_sp.shape[1]

                        if scores is not None:
                            scores = scores.view(-1, 1).repeat(1, k)
                            scores_sp = torch.min(scores, scores_sp)

                            # [B, S, K]
                            scores_sp = scores_sp.view(batch_size, -1, k)

                            # [B, K]
                            scores_sp, _ = torch.max(scores_sp, dim=1)

            if arg2 is not None:
                sources, scores = arg2, None

                # XXX
                scores = hops.prior(rel)

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
                            z_scores, z_emb = self.hop(hop_rel_2d, sources_2d, None, mask_indices=mask_indices)
                        else:
                            z_scores, z_emb = self.hop(hop_rel_2d, None, sources_2d, mask_indices=mask_indices)

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
                            scores = torch.min(z_scores_1d, scores.view(-1, 1).repeat(1, k).view(-1))
                    else:
                        # [B * S, K]
                        if is_reversed:
                            res_po, _ = self.model.forward(hop_rel_2d, sources_2d, None, mask_indices=mask_indices)
                        else:
                            _, res_po = self.model.forward(hop_rel_2d, None, sources_2d, mask_indices=mask_indices)

                        scores_po, subs_po = res_po
                        k = scores_po.shape[1]

                        if scores is not None:
                            scores = scores.view(-1, 1).repeat(1, k)
                            scores_po = torch.min(scores, scores_po)

                            # [B, S, K]
                            scores_po = scores_po.view(batch_size, -1, k)
                            # [B, K]
                            scores_po, _ = torch.max(scores_po, dim=1)

            if scores_sp is None and scores_po is None:
                scores_sp, scores_po = self.model.forward(rel, arg1, arg2, mask_indices=mask_indices)

            global_scores_sp = scores_sp if global_scores_sp is None else torch.max(global_scores_sp, scores_sp)
            global_scores_po = scores_po if global_scores_po is None else torch.max(global_scores_po, scores_po)

        if global_scores_sp is None and global_scores_po is None:
            global_scores_sp, global_scores_po = self.model.forward(rel, arg1, arg2, mask_indices=mask_indices)

        return global_scores_sp, global_scores_po

    def forward_(self,
                 rel: Tensor,
                 arg1: Optional[Tensor],
                 arg2: Optional[Tensor],
                 mask_indices: Optional[Tensor] = None,
                 *args, **kwargs) -> Tuple[Optional[Tuple[Tensor, Optional[Tensor]]],
                                           Optional[Tuple[Tensor, Optional[Tensor]]]]:
        batch_size, embedding_size = rel.shape[0], rel.shape[1]
        scores_sp = scores_po = None

        global_scores_sp = global_scores_po = None

        for hops, is_reversed in self.hops_lst:
            hop_rel_lst = hops(rel)
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
                        if is_reversed:
                            z_scores, z_emb = self.hop(hop_rel_2d, None, sources_2d, mask_indices=mask_indices)
                        else:
                            z_scores, z_emb = self.hop(hop_rel_2d, sources_2d, None, mask_indices=mask_indices)

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
                            scores = torch.min(z_scores_1d, scores.view(-1, 1).repeat(1, k).view(-1))
                    else:
                        # [B * S, K]
                        if is_reversed:
                            _, res_sp = self.model.forward_(hop_rel_2d, None, sources_2d, mask_indices=mask_indices)
                        else:
                            res_sp, _ = self.model.forward_(hop_rel_2d, sources_2d, None, mask_indices=mask_indices)

                        scores_sp, subs_sp = res_sp
                        k = scores_sp.shape[1]

                        if scores is not None:
                            scores = scores.view(-1, 1).repeat(1, k)
                            scores_sp = torch.min(scores, scores_sp)

                            # [B, S, K]
                            scores_sp = scores_sp.view(batch_size, -1, k)

                            # [B, K]
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
                        if is_reversed:
                            z_scores, z_emb = self.hop(hop_rel_2d, sources_2d, None, mask_indices=mask_indices)
                        else:
                            z_scores, z_emb = self.hop(hop_rel_2d, None, sources_2d, mask_indices=mask_indices)

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
                            scores = torch.min(z_scores_1d, scores.view(-1, 1).repeat(1, k).view(-1))
                    else:
                        # [B * S, K]
                        if is_reversed:
                            res_po, _ = self.model.forward_(hop_rel_2d, sources_2d, None, mask_indices=mask_indices)
                        else:
                            _, res_po = self.model.forward_(hop_rel_2d, None, sources_2d, mask_indices=mask_indices)

                        scores_po, subs_po = res_po
                        k = scores_po.shape[1]

                        if scores is not None:
                            scores = scores.view(-1, 1).repeat(1, k)
                            scores_po = torch.min(scores, scores_po)

                            # [B, S, K]
                            scores_po = scores_po.view(batch_size, -1, k)
                            # [B, K]
                            scores_po, _ = torch.max(scores_po, dim=1)

            if scores_sp is None and scores_po is None:
                (scores_sp, _), (scores_po, _) = self.model.forward_(rel, arg1, arg2, mask_indices=mask_indices)

            global_scores_sp = scores_sp if global_scores_sp is None else torch.max(global_scores_sp, scores_sp)
            global_scores_po = scores_po if global_scores_po is None else torch.max(global_scores_po, scores_po)

        if global_scores_sp is None and global_scores_po is None:
            (global_scores_sp, _), (global_scores_po, _) = self.model.forward_(rel, arg1, arg2, mask_indices=mask_indices)

        return (global_scores_sp, None), (global_scores_po, None)

    def forward__(self,
                  rel: Tensor,
                  arg1: Optional[Tensor],
                  arg2: Optional[Tensor],
                  mask_indices: Optional[Tensor] = None,
                  *args, **kwargs) -> Tuple[Optional[Tuple[Tensor, Optional[Tensor]]],
                                            Optional[Tuple[Tensor, Optional[Tensor]]]]:
        batch_size, embedding_size = rel.shape[0], rel.shape[1]
        scores_sp = scores_po = None

        global_scores_sp = global_scores_po = None

        for hops, is_reversed in self.hops_lst:
            hop_rel_lst = hops(rel)
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
                        if is_reversed:
                            z_scores, z_emb = self.hop(hop_rel_2d, None, sources_2d, mask_indices=mask_indices)
                        else:
                            z_scores, z_emb = self.hop(hop_rel_2d, sources_2d, None, mask_indices=mask_indices)

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
                            scores = torch.min(z_scores_1d, scores.view(-1, 1).repeat(1, k).view(-1))
                    else:
                        # [B * S, K]
                        if is_reversed:
                            _, res_sp = self.model.forward__(hop_rel_2d, None, sources_2d, mask_indices=mask_indices)
                        else:
                            res_sp, _ = self.model.forward__(hop_rel_2d, sources_2d, None, mask_indices=mask_indices)

                        scores_sp, subs_sp = res_sp
                        k = scores_sp.shape[1]

                        if scores is not None:
                            scores = scores.view(-1, 1).repeat(1, k)
                            scores_sp = torch.min(scores, scores_sp)

                            # [B, S, K]
                            scores_sp = scores_sp.view(batch_size, -1, k)

                            # [B, K]
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
                        if is_reversed:
                            z_scores, z_emb = self.hop(hop_rel_2d, sources_2d, None, mask_indices=mask_indices)
                        else:
                            z_scores, z_emb = self.hop(hop_rel_2d, None, sources_2d, mask_indices=mask_indices)

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
                            scores = torch.min(z_scores_1d, scores.view(-1, 1).repeat(1, k).view(-1))
                    else:
                        # [B * S, K]
                        if is_reversed:
                            res_po, _ = self.model.forward__(hop_rel_2d, sources_2d, None, mask_indices=mask_indices)
                        else:
                            _, res_po = self.model.forward__(hop_rel_2d, None, sources_2d, mask_indices=mask_indices)

                        scores_po, subs_po = res_po
                        k = scores_po.shape[1]

                        if scores is not None:
                            scores = scores.view(-1, 1).repeat(1, k)
                            scores_po = torch.min(scores, scores_po)

                            # [B, S, K]
                            scores_po = scores_po.view(batch_size, -1, k)
                            # [B, K]
                            scores_po, _ = torch.max(scores_po, dim=1)

            if scores_sp is None and scores_po is None:
                (scores_sp, _), (scores_po, _) = self.model.forward__(rel, arg1, arg2, mask_indices=mask_indices)

            global_scores_sp = scores_sp if global_scores_sp is None else torch.max(global_scores_sp, scores_sp)
            global_scores_po = scores_po if global_scores_po is None else torch.max(global_scores_po, scores_po)

        if global_scores_sp is None and global_scores_po is None:
            (global_scores_sp, _), (global_scores_po, _) = self.model.forward__(rel, arg1, arg2,
                                                                                mask_indices=mask_indices)

        return (global_scores_sp, None), (global_scores_po, None)

    def factor(self,
               embedding_vector: Tensor) -> Tensor:
        return self.model.factor(embedding_vector)
