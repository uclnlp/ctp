# -*- coding: utf-8 -*-

import torch
from torch import Tensor

from typing import Tuple, Callable


def do_merge(a_res: Tuple[Tensor, Tensor],
             b_res: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
    # [B, K], [B, K, E]
    a_scores, a_emb = a_res
    # [B, K], [B, K, E]
    b_scores, b_emb = b_res

    batch_size = a_scores.shape[0]
    k = a_scores.shape[1]
    emb_size = a_emb.shape[2]

    # [B, 2K]
    scores = torch.cat([
        a_scores.view(batch_size, k),
        b_scores.view(batch_size, -1)
    ], dim=1)

    # [B, 2K, E]
    emb = torch.cat([
        a_emb.view(batch_size, k, emb_size),
        b_emb.view(batch_size, -1, emb_size)
    ], dim=1)

    return scores, emb


def do_top(a_res: Tuple[Tensor, Tensor],
           b_res: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
    # [B, K], [B, K, E]
    a_scores, a_emb = a_res
    # [B, K], [B, K, E]
    b_scores, b_emb = b_res

    batch_size = a_scores.shape[0]
    k = a_scores.shape[1]
    emb_size = a_emb.shape[2]

    # [B, 2K]
    scores = torch.cat([
        a_scores.view(batch_size, k),
        b_scores.view(batch_size, k)
    ], dim=1)

    # [B, 2K, E]
    emb = torch.cat([
        a_emb.view(batch_size, k, emb_size),
        b_emb.view(batch_size, k, emb_size)
    ], dim=1)

    # [B, K]
    _scores, _ind = torch.topk(scores, k=k, dim=1)

    ind_3d = _ind.view(batch_size, k, 1)
    _emb = torch.gather(emb, 1, ind_3d.repeat(1, 1, emb_size))

    return _scores, _emb


def do_cmp(a_res: Tuple[Tensor, Tensor],
           b_res: Tuple[Tensor, Tensor],
           op: Callable = torch.max) -> Tuple[Tensor, Tensor]:
    # [B, K], [B, K, E]
    a_scores, a_emb = a_res
    # [B, K], [B, K, E]
    b_scores, b_emb = b_res

    batch_size = a_scores.shape[0]
    k = a_scores.shape[1]
    emb_size = a_emb.shape[2]

    cat = torch.cat([
        a_scores.reshape(batch_size, k, 1),
        b_scores.reshape(batch_size, k, 1)], dim=2)
    new_scores, ind = op(cat, dim=2)

    ind_3d = ind.view(batch_size, k, 1).repeat(1, 1, emb_size)
    new_emb = torch.where(ind_3d == 0, a_emb, b_emb)

    return new_scores, new_emb
