# -*- coding: utf-8 -*-

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from ctp.clutrr.models.kb import BatchNeuralKB
from ctp.clutrr.models.util import uniform

from ctp.reformulators import BaseReformulator
from ctp.reformulators import GNTPReformulator

from typing import Tuple, Optional, List

import logging

logger = logging.getLogger(__name__)


class BatchUnary(nn.Module):
    def __init__(self,
                 model: BatchNeuralKB,
                 hops_lst: List[Tuple[BaseReformulator, bool]],
                 k: int = 10,
                 tnorm_name: str = 'min',
                 R: Optional[int] = None):
        super().__init__()

        self.model: BatchNeuralKB = model
        self.k = k

        self.tnorm_name = tnorm_name
        assert self.tnorm_name in {'min', 'prod', 'mean'}

        self.R = R

        self._hops_lst = nn.ModuleList([hops for hops, _ in hops_lst])
        self.hops_lst = hops_lst

        logger.info(f'BatchUnary(k={k}, hops_lst={[h.__class__.__name__ for h in self._hops_lst]})')

    def _tnorm(self, x: Tensor, y: Tensor) -> Tensor:
        res = None
        if self.tnorm_name == 'min':
            res = torch.min(x, y)
        elif self.tnorm_name == 'prod':
            res = x * y
        elif self.tnorm_name == 'mean':
            res = (x + y) / 2
        assert res is not None
        return res

    def r_hop(self,
              rel: Tensor, arg1: Optional[Tensor], arg2: Optional[Tensor],
              facts: List[Tensor],
              nb_facts: Tensor,
              entity_embeddings: Tensor,
              nb_entities: Tensor) -> Tuple[Tensor, Tensor]:
        assert (arg1 is None) ^ (arg2 is None)

        batch_size, embedding_size = rel.shape[0], rel.shape[1]

        # [B, N]
        scores_sp, scores_po = self.model.forward(rel, arg1, arg2, facts, nb_facts, entity_embeddings, nb_entities)
        scores = scores_sp if arg2 is None else scores_po

        k = min(self.k, scores.shape[1])

        # [B, K], [B, K]
        z_scores, z_indices = torch.topk(scores, k=k, dim=1)

        dim_1 = torch.arange(z_scores.shape[0], device=z_scores.device).view(-1, 1).repeat(1, k).view(-1)
        dim_2 = z_indices.view(-1)

        entity_embeddings, _ = uniform(z_scores, entity_embeddings)

        z_emb = entity_embeddings[dim_1, dim_2].view(z_scores.shape[0], k, -1)

        assert z_emb.shape[0] == batch_size
        assert z_emb.shape[2] == embedding_size

        return z_scores, z_emb

    def score(self,
              rel: Tensor, arg1: Tensor, arg2: Tensor,
              facts: List[Tensor], nb_facts: Tensor,
              entity_embeddings: Tensor, nb_entities: Tensor) -> Tensor:
        res = self.depth_r_score(rel, arg1, arg2, facts, nb_facts, entity_embeddings, nb_entities, depth=1)
        return res

    def depth_r_score(self,
                      rel: Tensor, arg1: Tensor, arg2: Tensor,
                      facts: List[Tensor],
                      nb_facts: Tensor,
                      entity_embeddings: Tensor,
                      nb_entities: Tensor,
                      depth: int) -> Tensor:
        assert depth == 1

        batch_size, embedding_size = rel.shape[0], rel.shape[1]
        global_res = None

        mask = None

        new_hops_lst = self.hops_lst

        if self.R is not None:
            batch_rules_scores = torch.cat([h.prior(rel).view(-1, 1) for h, _ in self.hops_lst], 1)
            topk, indices = torch.topk(batch_rules_scores, self.R)

            # [R x E]
            rule_heads = torch.cat([h.head for h, _ in self.hops_lst], dim=0)
            rule_body1s = torch.cat([h.memory_lst[0] for h, _ in self.hops_lst], dim=0)
            rule_body2s = torch.cat([h.memory_lst[1] for h, _ in self.hops_lst], dim=0)

            kernel = self.hops_lst[0][0].kernel
            new_rule_heads = F.embedding(indices, rule_heads)
            new_rule_body1s = F.embedding(indices, rule_body1s)
            new_rule_body2s = F.embedding(indices, rule_body2s)

            # print(new_rule_heads.shape[1], self.R)
            assert new_rule_heads.shape[1] == self.R

            new_hops_lst = []
            for i in range(new_rule_heads.shape[1]):
                r = GNTPReformulator(kernel=kernel, head=new_rule_heads[:, i, :],
                                     body=[new_rule_body1s[:, i, :], new_rule_body2s[:, i, :]])
                new_hops_lst += [(r, False)]

        for rule_idx, (hops_generator, is_reversed) in enumerate(new_hops_lst):
            sources, scores = arg1, None

            # XXX
            prior = hops_generator.prior(rel)
            if prior is not None:

                if mask is not None:
                    prior = prior * mask[:, rule_idx]
                    if (prior != 0.0).sum() == 0:
                        continue

                scores = prior

            hop_rel_lst = hops_generator(rel)

            for hop_idx, hop_rel in enumerate(hop_rel_lst, start=1):
                # [B * S, E]
                sources_2d = sources.view(-1, embedding_size)
                nb_sources = sources_2d.shape[0]

                nb_branches = nb_sources // batch_size

                hop_rel_3d = hop_rel.view(-1, 1, embedding_size).repeat(1, nb_branches, 1)
                hop_rel_2d = hop_rel_3d.view(-1, embedding_size)

                # [B * S, K], [B * S, K, E]
                if is_reversed:
                    z_scores, z_emb = self.r_hop(hop_rel_2d, None, sources_2d, facts, nb_facts, entity_embeddings, nb_entities)
                else:
                    z_scores, z_emb = self.r_hop(hop_rel_2d, sources_2d, None, facts, nb_facts, entity_embeddings, nb_entities)
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

            if scores is not None:
                scores_2d = scores.view(batch_size, -1)
                res, _ = torch.max(scores_2d, dim=1)
            else:
                assert False
                # res = self.model.score(rel, arg1, arg2, facts=facts, nb_facts=nb_facts,
                #                        entity_embeddings=entity_embeddings, nb_entities=nb_entities)

            global_res = res if global_res is None else torch.max(global_res, res)

        return global_res

    def factor(self,
               embedding_vector: Tensor) -> Tensor:
        return self.model.factor(embedding_vector)

    def extra_factors(self,
                      rel: Tensor, arg1: Optional[Tensor], arg2: Optional[Tensor]) -> List[Tensor]:
        return [hop_generator(rel) for hop_generators in self.hops_lst for hop_generator in hop_generators]
