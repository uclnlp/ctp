# -*- coding: utf-8 -*-

import numpy as np
# from tqdm import tqdm

import torch
from torch import nn

from kbcr.util import make_batches
from kbcr.models import BaseLatentFeatureModel

from typing import Tuple, Dict


def evaluate_slow(entity_embeddings: nn.Embedding,
                  predicate_embeddings: nn.Embedding,
                  test_triples: Tuple[str, str, str],
                  all_triples: Tuple[str, str, str],
                  entity_to_index: Dict[str, int],
                  predicate_to_index: Dict[str, int],
                  model: BaseLatentFeatureModel,
                  batch_size: int,
                  device: torch.device):

    xs = np.array([entity_to_index.get(s) for (s, _, _) in test_triples])
    xp = np.array([predicate_to_index.get(p) for (_, p, _) in test_triples])
    xo = np.array([entity_to_index.get(o) for (_, _, o) in test_triples])

    sp_to_o, po_to_s = {}, {}
    for s, p, o in all_triples:
        s_idx, p_idx, o_idx = entity_to_index.get(s), predicate_to_index.get(p), entity_to_index.get(o)
        sp_key = (s_idx, p_idx)
        po_key = (p_idx, o_idx)

        if sp_key not in sp_to_o:
            sp_to_o[sp_key] = []
        if po_key not in po_to_s:
            po_to_s[po_key] = []

        sp_to_o[sp_key] += [o_idx]
        po_to_s[po_key] += [s_idx]

    assert xs.shape == xp.shape == xo.shape
    nb_test_triples = xs.shape[0]

    batches = make_batches(nb_test_triples, batch_size)

    hits = dict()
    hits_at = [1, 3, 5, 10]

    for hits_at_value in hits_at:
        hits[hits_at_value] = 0.0

    def hits_at_n(n_, rank):
        if rank <= n_:
            hits[n_] = hits.get(n_, 0) + 1

    counter = 0
    mrr = 0.0

    ranks_l, ranks_r = [], []
    for start, end in batches:
        batch_xs = xs[start:end]
        batch_xp = xp[start:end]
        batch_xo = xo[start:end]

        batch_size = batch_xs.shape[0]
        counter += batch_size * 2

        with torch.no_grad():
            tensor_xs = torch.from_numpy(batch_xs).to(device)
            tensor_xp = torch.from_numpy(batch_xp).to(device)
            tensor_xo = torch.from_numpy(batch_xo).to(device)

            tensor_xs_emb = entity_embeddings(tensor_xs)
            tensor_xp_emb = predicate_embeddings(tensor_xp)
            tensor_xo_emb = entity_embeddings(tensor_xo)
            # print(entity_embeddings.weight.shape)

            if model.model.facts[0].shape[0] < 90000:
                res_sp, res_po = model.forward_(tensor_xp_emb, tensor_xs_emb, tensor_xo_emb)
            else:
                res_sp, res_po = model.forward__(tensor_xp_emb, tensor_xs_emb, tensor_xo_emb)

            _scores_sp, _ = res_sp
            _scores_po, _ = res_po

            scores_sp, scores_po = _scores_sp.cpu().numpy(), _scores_po.cpu().numpy()

            del _scores_sp, _scores_po
            del tensor_xs, tensor_xp, tensor_xo
            del tensor_xs_emb, tensor_xp_emb, tensor_xo_emb
            del res_sp, res_po
            # print(scores_sp.shape, scores_po.shape)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        batch_size = batch_xs.shape[0]
        for elem_idx in range(batch_size):
            s_idx, p_idx, o_idx = batch_xs[elem_idx], batch_xp[elem_idx], batch_xo[elem_idx]

            # Code for the filtered setting
            sp_key = (s_idx, p_idx)
            po_key = (p_idx, o_idx)

            o_to_remove = sp_to_o[sp_key]
            s_to_remove = po_to_s[po_key]

            for tmp_o_idx in o_to_remove:
                if tmp_o_idx != o_idx:
                    scores_sp[elem_idx, tmp_o_idx] = - np.infty

            for tmp_s_idx in s_to_remove:
                if tmp_s_idx != s_idx:
                    scores_po[elem_idx, tmp_s_idx] = - np.infty
            # End of code for the filtered setting

            rank_l = 1 + np.argsort(np.argsort(- scores_po[elem_idx, :]))[s_idx]
            rank_r = 1 + np.argsort(np.argsort(- scores_sp[elem_idx, :]))[o_idx]

            ranks_l += [rank_l]
            ranks_r += [rank_r]

            mrr += 1.0 / rank_l
            mrr += 1.0 / rank_r

            for n in hits_at:
                hits_at_n(n, rank_l)

            for n in hits_at:
                hits_at_n(n, rank_r)

    counter = float(counter)

    mrr /= counter

    for n in hits_at:
        hits[n] /= counter

    metrics = dict()
    metrics['MRR'] = mrr
    for n in hits_at:
        metrics['hits@{}'.format(n)] = hits[n]

    return metrics
