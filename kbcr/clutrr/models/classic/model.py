# -*- coding: utf-8 -*-

import torch
from torch import nn, Tensor

from kbcr.clutrr.models.classic.kb import NeuralKB
from kbcr.reformulators import BaseReformulator

from typing import Tuple, Optional, List, Generator

import logging

logger = logging.getLogger(__name__)


class Hoppy(nn.Module):
    def __init__(self,
                 model: NeuralKB,
                 hops_lst: List[Tuple[BaseReformulator, bool]],
                 max_depth: int = 0):
        super().__init__()
        self.model = model
        self.max_depth = max_depth

        self._hops_lst = nn.ModuleList([hops for hops, _ in hops_lst])
        self.hops_lst = hops_lst

    def score(self,
              rel: Tensor,
              arg1: Optional[Tensor],
              arg2: Optional[Tensor],
              facts: List[Tensor]) -> Tensor:
        batch_size, res = rel.shape[0], None
        for scores, _ in self.or_(rel, arg1, arg2, facts, self.max_depth):
            tmp, _ = scores.view(batch_size, -1).max(dim=1)
            res = tmp if res is None else torch.max(res, tmp)
        return res

    def or_(self,
            rel: Tensor,
            arg1: Optional[Tensor],
            arg2: Optional[Tensor],
            facts: List[Tensor],
            depth: int) -> Generator[Tuple[Tensor, Optional[Tensor]], None, None]:
        # First, unify with the facts
        ground_res = self.model(rel, arg1, arg2, facts)
        yield ground_res

        # Iterate over the reformulators and generate the rule bodies, which are answered by the AND module
        for hops_generator, is_reversed in self.hops_lst:
            # Generate the rule body, e.g. [parent, parent]:
            hops_lst = hops_generator(rel)
            _arg1, _arg2 = (arg1, arg2) if not is_reversed else (arg2, arg1)
            _hops_lst = hops_lst if not is_reversed else reversed(hops_lst)

            for res in self.and_(_arg1, _arg2, facts, depth - 1, _hops_lst):
                yield res

    def and_(self,
             arg1: Optional[Tensor],
             arg2: Optional[Tensor],
             facts: List[Tensor],
             depth: int,
             hops_lst: List[Tensor]) -> Generator[Tuple[Tensor, Optional[Tensor]], None, None]:

        if depth < 0 or not hops_lst:
            return

        def reshape_emb(emb: Optional[Tensor], nb_entries: int) -> Optional[Tensor]:
            if emb is None:
                return None
            batch_size, emb_size = emb.shape[0], emb.shape[1]
            mult = nb_entries // batch_size
            emb = emb.view(batch_size, 1, emb_size)
            emb = emb.repeat(1, mult, 1)
            emb = emb.view(-1, emb_size)
            return emb

        def reshape_scores(scores: Optional[Tensor], nb_entries: int) -> Optional[Tensor]:
            mult = nb_entries // scores.shape[0]
            return scores.view(scores.shape[0], 1).repeat(1, mult).view(-1)

        # Prove the current body atom:
        cur_hop = reshape_emb(hops_lst[0], arg1.shape[0])
        rem_hop_lst = hops_lst[1:]

        if len(rem_hop_lst) > 0:
            # [B, K], [B, K, E]
            for scores, subs in self.or_(cur_hop, arg1, None, facts, depth):
                embedding_size = subs.shape[1]

                arg1_ = subs.view(-1, embedding_size)
                arg2_ = arg2 if arg2 is None else reshape_emb(arg2, arg1_.shape[0])

                if arg2 is not None:
                    assert arg1_.shape[0] == arg2_.shape[0]

                for res in self.and_(arg1_, arg2_, facts, depth, rem_hop_lst):
                    nb_entries = max(scores.shape[0], res[0].shape[0])

                    scores_ = torch.min(reshape_scores(res[0], nb_entries), reshape_scores(scores, nb_entries))
                    subs_ = reshape_emb(res[1], nb_entries)

                    yield scores_, subs_
        else:
            # [B, K], [B, K, E]
            for scores, subs in self.or_(cur_hop, arg1, arg2, facts, depth):
                yield scores, subs
