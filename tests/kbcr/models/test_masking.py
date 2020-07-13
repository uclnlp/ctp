# -*- coding: utf-8 -*-

import numpy as np

import torch
from torch import nn

from kbcr.kernels import GaussianKernel
from kbcr.models import NeuralKB
from kbcr.models.reasoning import SimpleHoppy
from kbcr.reformulators import SymbolicReformulator

import pytest


@pytest.mark.light
def test_masking_v1():
    nb_entities = 10
    nb_predicates = 5
    embedding_size = 10

    init_size = 1.0

    rs = np.random.RandomState(0)

    for _ in range(1):
        for position in [0, 1]:
            for st in ['min', 'concat']:
                with torch.no_grad():
                    triples = [
                        ('a', 'p', 'b'),
                        ('c', 'q', 'd')
                    ]
                    entity_to_index = {'a': 0, 'b': 1, 'c': 2, 'd': 3}
                    predicate_to_index = {'p': 0, 'q': 1}

                    kernel = GaussianKernel()

                    entity_embeddings = nn.Embedding(nb_entities, embedding_size * 2, sparse=True)
                    predicate_embeddings = nn.Embedding(nb_predicates, embedding_size * 2, sparse=True)

                    entity_embeddings.weight.data *= init_size
                    predicate_embeddings.weight.data *= init_size

                    fact_rel = torch.from_numpy(np.array([predicate_to_index[p] for (_, p, _) in triples]))
                    fact_arg1 = torch.from_numpy(np.array([entity_to_index[s] for (s, _, _) in triples]))
                    fact_arg2 = torch.from_numpy(np.array([entity_to_index[o] for (_, _, o) in triples]))
                    facts = [fact_rel, fact_arg1, fact_arg2]

                    model = NeuralKB(entity_embeddings=entity_embeddings, predicate_embeddings=predicate_embeddings,
                                     kernel=kernel, facts=facts, scoring_type=st)

                    xs_np = rs.randint(nb_entities, size=32)
                    xp_np = rs.randint(nb_predicates, size=32)
                    xo_np = rs.randint(nb_entities, size=32)
                    xi_np = np.array([position] * xs_np.shape[0])

                    xs_np[0] = 0
                    xp_np[0] = 0
                    xo_np[0] = 1

                    xs_np[1] = 2
                    xp_np[1] = 1
                    xo_np[1] = 3

                    xs = torch.from_numpy(xs_np)
                    xp = torch.from_numpy(xp_np)
                    xo = torch.from_numpy(xo_np)
                    xi = torch.from_numpy(xi_np)

                    xs_emb = entity_embeddings(xs)
                    xp_emb = predicate_embeddings(xp)
                    xo_emb = entity_embeddings(xo)

                    model.mask_indices = xi

                    scores = model.forward(xp_emb, xs_emb, xo_emb)
                    inf = model.score(xp_emb, xs_emb, xo_emb)

                    if position == 0:
                        assert inf[0] < 0.5
                        assert inf[1] > 0.9
                    elif position == 1:
                        assert inf[0] > 0.9
                        assert inf[1] < 0.5

                    scores_sp, scores_po = scores

                    inf = inf.cpu().numpy()
                    scores_sp = scores_sp.cpu().numpy()
                    scores_po = scores_po.cpu().numpy()

                    for i in range(xs.shape[0]):
                        np.testing.assert_allclose(inf[i], scores_sp[i, xo[i]], rtol=1e-5, atol=1e-5)
                        np.testing.assert_allclose(inf[i], scores_po[i, xs[i]], rtol=1e-5, atol=1e-5)


@pytest.mark.light
def test_masking_v2():
    nb_entities = 10
    nb_predicates = 5
    embedding_size = 10

    rs = np.random.RandomState(0)

    for _ in range(1):
        for position in [0, 1, 2]:
            for st in ['min', 'concat']:
                with torch.no_grad():
                    triples = [
                        ('a', 'p', 'b'),
                        ('b', 'q', 'c'),
                        ('a', 'p', 'c')
                    ]
                    entity_to_index = {'a': 0, 'b': 1, 'c': 2, 'd': 3}
                    predicate_to_index = {'p': 0, 'q': 1}

                    kernel = GaussianKernel()

                    entity_emb = nn.Embedding(nb_entities, embedding_size * 2, sparse=True)
                    predicate_emb = nn.Embedding(nb_predicates, embedding_size * 2, sparse=True)

                    fact_rel = torch.from_numpy(np.array([predicate_to_index[p] for (_, p, _) in triples]))
                    fact_arg1 = torch.from_numpy(np.array([entity_to_index[s] for (s, _, _) in triples]))
                    fact_arg2 = torch.from_numpy(np.array([entity_to_index[o] for (_, _, o) in triples]))
                    facts = [fact_rel, fact_arg1, fact_arg2]

                    base = NeuralKB(entity_embeddings=entity_emb, predicate_embeddings=predicate_emb,
                                    kernel=kernel, facts=facts, scoring_type=st)

                    indices = torch.from_numpy(np.array([predicate_to_index['p'], predicate_to_index['q']]))
                    reformulator = SymbolicReformulator(predicate_emb, indices)
                    model = SimpleHoppy(base, entity_emb, hops=reformulator)

                    xs_np = rs.randint(nb_entities, size=32)
                    xp_np = rs.randint(nb_predicates, size=32)
                    xo_np = rs.randint(nb_entities, size=32)
                    xi_np = np.array([position] * xs_np.shape[0])

                    xs_np[0] = 0
                    xp_np[0] = 0
                    xo_np[0] = 1

                    xs_np[1] = 1
                    xp_np[1] = 1
                    xo_np[1] = 2

                    xs_np[2] = 0
                    xp_np[2] = 0
                    xo_np[2] = 2

                    xs = torch.from_numpy(xs_np)
                    xp = torch.from_numpy(xp_np)
                    xo = torch.from_numpy(xo_np)
                    xi = torch.from_numpy(xi_np)

                    xs_emb = entity_emb(xs)
                    xp_emb = predicate_emb(xp)
                    xo_emb = entity_emb(xo)

                    # xi = None
                    base.mask_indices = xi

                    scores = model.forward(xp_emb, xs_emb, xo_emb)
                    inf = model.score(xp_emb, xs_emb, xo_emb)

                    if position in {0, 1}:
                        assert inf[2] < 0.5
                    else:
                        assert inf[2] > 0.9

                    scores_sp, scores_po = scores

                    inf = inf.cpu().numpy()
                    scores_sp = scores_sp.cpu().numpy()
                    scores_po = scores_po.cpu().numpy()

                    for i in range(xs.shape[0]):
                        np.testing.assert_allclose(inf[i], scores_sp[i, xo[i]], rtol=1e-5, atol=1e-5)
                        np.testing.assert_allclose(inf[i], scores_po[i, xs[i]], rtol=1e-5, atol=1e-5)


if __name__ == '__main__':
    pytest.main([__file__])
