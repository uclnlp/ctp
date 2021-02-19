# -*- coding: utf-8 -*-

import numpy as np

import torch
from torch import nn, Tensor

from ctp.kernels import GaussianKernel
from ctp.clutrr.models import BatchNeuralKB

from typing import List, Dict, Tuple, Optional

import pytest


def encode_relation(facts: List[Tuple[str, str, str]],
                    relation_embeddings: nn.Embedding,
                    relation_to_idx: Dict[str, int],
                    device: Optional[torch.device] = None) -> Tensor:
    indices_np = np.array([relation_to_idx[r] for _, r, _ in facts], dtype=np.int64)
    indices = torch.from_numpy(indices_np)
    if device is not None:
        indices = indices.to(device)
    return relation_embeddings(indices)


def encode_arguments(facts: List[Tuple[str, str, str]],
                     entity_embeddings: nn.Embedding,
                     entity_to_idx: Dict[str, int],
                     device: Optional[torch.device] = None) -> Tuple[Tensor, Tensor]:
    indices_np = np.array([[entity_to_idx[s], entity_to_idx[o]] for s, _, o in facts], dtype=np.int64)
    indices = torch.from_numpy(indices_np)
    if device is not None:
        indices = indices.to(device)
    emb = entity_embeddings(indices)
    return emb[:, 0, :], emb[:, 1, :]


@pytest.mark.light
def test_batch_v1():
    embedding_size = 100

    triples = [('a', 'p', f'b{i}') for i in range(128)]

    entity_lst = sorted({e for (e, _, _) in triples} | {e for (e, _, e) in triples})
    predicate_lst = sorted({p for (_, p, _) in triples})

    nb_entities, nb_predicates = len(entity_lst), len(predicate_lst)

    entity_to_index = {e: i for i, e in enumerate(entity_lst)}
    predicate_to_index = {p: i for i, p in enumerate(predicate_lst)}

    kernel = GaussianKernel()

    entity_embeddings = nn.Embedding(nb_entities, embedding_size, sparse=True)
    predicate_embeddings = nn.Embedding(nb_predicates, embedding_size, sparse=True)

    for scoring_type in ['concat']:  # ['min', 'concat']:
        for _fact_size in range(len(triples)):
            with torch.no_grad():
                model = BatchNeuralKB(kernel=kernel, scoring_type=scoring_type)

                xp_emb = encode_relation(facts=triples, relation_embeddings=predicate_embeddings,
                                         relation_to_idx=predicate_to_index)
                xs_emb, xo_emb = encode_arguments(facts=triples, entity_embeddings=entity_embeddings,
                                                  entity_to_idx=entity_to_index)

                batch_size = len(triples)
                fact_size = len(triples)

                rel_emb = xp_emb.view(1, fact_size, -1).repeat(batch_size, 1, 1)
                arg1_emb = xs_emb.view(1, fact_size, -1).repeat(batch_size, 1, 1)
                arg2_emb = xo_emb.view(1, fact_size, -1).repeat(batch_size, 1, 1)

                nb_facts = torch.tensor([_fact_size for _ in range(batch_size)], dtype=torch.long)

                facts = [rel_emb, arg1_emb, arg2_emb]

                inf = model.score(xp_emb, xs_emb, xo_emb, facts=facts, nb_facts=nb_facts)
                inf_np = inf.cpu().numpy()

                exp = [1] * _fact_size + [0] * (batch_size - _fact_size)
                np.testing.assert_allclose(inf_np, exp, rtol=1e-2, atol=1e-2)

                print(inf_np)


if __name__ == '__main__':
    pytest.main([__file__])
    # test_batch_v1()
