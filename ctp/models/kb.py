# -*- coding: utf-8 -*-

import torch
from torch import nn, Tensor

from ctp.kernels import BaseKernel
from ctp.models.base import BaseLatentFeatureModel
from ctp.models.util.masking import generate_kb_mask

from typing import List, Tuple, Optional

import logging

logger = logging.getLogger(__name__)


class NeuralKB(BaseLatentFeatureModel):
    def __init__(self,
                 entity_embeddings: nn.Embedding,
                 predicate_embeddings: nn.Embedding,
                 facts: List[Tensor],
                 kernel: BaseKernel,
                 scoring_type: str = 'concat') -> None:
        super().__init__()

        self.entity_embeddings = entity_embeddings
        self.predicate_embeddings = predicate_embeddings

        self.facts = facts
        assert len(self.facts) == 3

        self.kernel = kernel

        self.scoring_type = scoring_type
        assert self.scoring_type in {'concat', 'min'}

        self._mask_indices = None

    @property
    def mask_indices(self) -> Optional[Tensor]:
        return self._mask_indices

    @mask_indices.setter
    def mask_indices(self,
                     value: Optional[Tensor]):
        self._mask_indices = value

    def generate_facts(self) -> List[Tensor]:
        fact_rel, fact_arg1, fact_arg2 = self.facts
        emb_rel = self.predicate_embeddings(fact_rel)
        emb_arg1 = self.entity_embeddings(fact_arg1)
        emb_arg2 = self.entity_embeddings(fact_arg2)
        return [emb_rel, emb_arg1, emb_arg2]

    def score(self,
              rel: Tensor,
              arg1: Tensor,
              arg2: Tensor,
              *args, **kwargs) -> Tensor:

        fact_emb_lst = self.generate_facts()

        nb_facts = fact_emb_lst[0].shape[0]
        nb_goals = rel.shape[0]

        if self.scoring_type in {'concat'}:
            fact_emb = torch.cat(fact_emb_lst, dim=1)
            batch_emb = torch.cat([rel, arg1, arg2], dim=1)
            # [B, KB]
            pairwise = self.kernel.pairwise(batch_emb, fact_emb)
        else:
            pairwise = None
            for a, b in zip([rel, arg1, arg2], fact_emb_lst):
                l_pairwise = self.kernel.pairwise(a, b)
                # [B, KB]
                pairwise = l_pairwise if pairwise is None else torch.min(pairwise, l_pairwise)

        if self.mask_indices is not None:
            repeat_mask = nb_goals // self.mask_indices.shape[0]
            mask_indices = self.mask_indices.view(-1, 1).repeat(1, repeat_mask).view(-1)
            mask = generate_kb_mask(indices=mask_indices, batch_size=nb_goals, kb_size=nb_facts)
            pairwise *= mask

        # [B]
        (res, _) = torch.max(pairwise, dim=1)
        return res

    def forward(self,
                rel: Tensor,
                arg1: Optional[Tensor],
                arg2: Optional[Tensor],
                *args, **kwargs) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        # [N, E]
        emb = self.entity_embeddings.weight

        batch_size = rel.shape[0]
        nb_entities = emb.shape[0]

        fact_emb_lst = self.generate_facts()
        nb_facts = fact_emb_lst[0].shape[0]

        score_sp_lst, score_po_lst = [], []

        mask_indices = None
        if self.mask_indices is not None:
            repeat_mask = batch_size // self.mask_indices.shape[0]
            mask_indices = self.mask_indices.view(-1, 1).repeat(1, repeat_mask).view(-1)

        for i in range(batch_size):
            i_emb_rel = rel[i, :].view(1, -1).repeat(nb_entities, 1)

            if arg1 is not None:
                # [N, E]
                i_emb_arg1 = arg1[i, :].view(1, -1).repeat(nb_entities, 1)

                if self.scoring_type in {'concat'}:
                    # [KB, 3E]
                    fact_emb = torch.cat(fact_emb_lst, dim=1)
                    # [N, 3E]
                    batch_emb = torch.cat([i_emb_rel, i_emb_arg1, emb], dim=1)

                    # [B, KB]
                    pairwise_sp = self.kernel.pairwise(batch_emb, fact_emb)
                else:
                    pairwise_sp = None
                    for a, b in zip([i_emb_rel, i_emb_arg1, emb], fact_emb_lst):
                        l_pairwise = self.kernel.pairwise(a, b)
                        # [B, KB]
                        pairwise_sp = l_pairwise if pairwise_sp is None else torch.min(pairwise_sp, l_pairwise)

                if mask_indices is not None:
                    indices = mask_indices[i].repeat(nb_entities)
                    mask = generate_kb_mask(indices=indices,
                                            batch_size=i_emb_rel.shape[0],
                                            kb_size=nb_facts)
                    pairwise_sp *= mask

                i_score_sp, _ = torch.max(pairwise_sp, dim=1)
                score_sp_lst += [i_score_sp.view(1, -1)]

            if arg2 is not None:
                i_emb_arg2 = arg2[i, :].view(1, -1).repeat(nb_entities, 1)

                if self.scoring_type in {'concat'}:
                    # [KB, 3E]
                    fact_emb = torch.cat(fact_emb_lst, dim=1)
                    # [N, 3E]
                    batch_emb = torch.cat([i_emb_rel, emb, i_emb_arg2], dim=1)

                    # [B, KB]
                    pairwise_po = self.kernel.pairwise(batch_emb, fact_emb)
                else:
                    pairwise_po = None
                    for a, b in zip([i_emb_rel, emb, i_emb_arg2], fact_emb_lst):
                        l_pairwise = self.kernel.pairwise(a, b)
                        # [B, KB]
                        pairwise_po = l_pairwise if pairwise_po is None else torch.min(pairwise_po, l_pairwise)

                if mask_indices is not None:
                    indices = mask_indices[i].repeat(nb_entities)
                    mask = generate_kb_mask(indices=indices,
                                            batch_size=i_emb_rel.shape[0],
                                            kb_size=nb_facts)
                    pairwise_po *= mask

                i_score_po, _ = torch.max(pairwise_po, dim=1)
                score_po_lst += [i_score_po.view(1, -1)]

        # [B, N]
        score_sp = torch.cat(score_sp_lst, dim=0) if arg1 is not None else None
        score_po = torch.cat(score_po_lst, dim=0) if arg2 is not None else None

        return score_sp, score_po

    def factor(self,
               embedding_vector: Tensor) -> Tensor:
        return embedding_vector
