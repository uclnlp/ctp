# -*- coding: utf-8 -*-

import numpy as np

import torch
from torch import nn

from ctp.kernels import GaussianKernel
from ctp.models import DistMult, ComplEx, NeuralKB
from ctp.models.reasoning import SimpleHoppy, RecursiveHoppy, Hoppy
from ctp.reformulators import LinearReformulator, AttentiveReformulator

import pytest


@pytest.mark.light
def test_distmult_v1():
    nb_entities = 10
    nb_predicates = 5
    embedding_size = 10

    init_size = 1.0

    rs = np.random.RandomState(0)

    for _ in range(128):
        with torch.no_grad():
            entity_embeddings = nn.Embedding(nb_entities, embedding_size, sparse=True)
            predicate_embeddings = nn.Embedding(nb_predicates, embedding_size, sparse=True)

            entity_embeddings.weight.data *= init_size
            predicate_embeddings.weight.data *= init_size

            model = DistMult(entity_embeddings)

            xs = torch.from_numpy(rs.randint(nb_entities, size=32))
            xp = torch.from_numpy(rs.randint(nb_predicates, size=32))
            xo = torch.from_numpy(rs.randint(nb_entities, size=32))

            xs_emb = entity_embeddings(xs)
            xp_emb = predicate_embeddings(xp)
            xo_emb = entity_embeddings(xo)

            scores = model.forward(xp_emb, xs_emb, xo_emb)
            inf = model.score(xp_emb, xs_emb, xo_emb)

            scores_sp, scores_po = scores

            inf = inf.cpu().numpy()
            scores_sp = scores_sp.cpu().numpy()
            scores_po = scores_po.cpu().numpy()

            for i in range(xs.shape[0]):
                np.testing.assert_allclose(inf[i], scores_sp[i, xo[i]], rtol=1e-3, atol=1e-3)
                np.testing.assert_allclose(inf[i], scores_po[i, xs[i]], rtol=1e-3, atol=1e-3)


@pytest.mark.light
def test_complex_v1():
    nb_entities = 10
    nb_predicates = 5
    embedding_size = 10

    init_size = 1.0

    rs = np.random.RandomState(0)

    for _ in range(128):
        with torch.no_grad():
            entity_embeddings = nn.Embedding(nb_entities, embedding_size * 2, sparse=True)
            predicate_embeddings = nn.Embedding(nb_predicates, embedding_size * 2, sparse=True)

            entity_embeddings.weight.data *= init_size
            predicate_embeddings.weight.data *= init_size

            model = ComplEx(entity_embeddings)

            xs = torch.from_numpy(rs.randint(nb_entities, size=32))
            xp = torch.from_numpy(rs.randint(nb_predicates, size=32))
            xo = torch.from_numpy(rs.randint(nb_entities, size=32))

            xs_emb = entity_embeddings(xs)
            xp_emb = predicate_embeddings(xp)
            xo_emb = entity_embeddings(xo)

            scores = model.forward(xp_emb, xs_emb, xo_emb)
            inf = model.score(xp_emb, xs_emb, xo_emb)

            scores_sp, scores_po = scores

            inf = inf.cpu().numpy()
            scores_sp = scores_sp.cpu().numpy()
            scores_po = scores_po.cpu().numpy()

            for i in range(xs.shape[0]):
                np.testing.assert_allclose(inf[i], scores_sp[i, xo[i]], rtol=1e-3, atol=1e-3)
                np.testing.assert_allclose(inf[i], scores_po[i, xs[i]], rtol=1e-3, atol=1e-3)


@pytest.mark.light
def test_neuralkb_v1():
    nb_entities = 10
    nb_predicates = 5
    embedding_size = 10

    rs = np.random.RandomState(0)

    for _ in range(32):
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

                fact_rel = torch.from_numpy(np.array([predicate_to_index[p] for (_, p, _) in triples]))
                fact_arg1 = torch.from_numpy(np.array([entity_to_index[s] for (s, _, _) in triples]))
                fact_arg2 = torch.from_numpy(np.array([entity_to_index[o] for (_, _, o) in triples]))
                facts = [fact_rel, fact_arg1, fact_arg2]

                model = NeuralKB(entity_embeddings=entity_embeddings, predicate_embeddings=predicate_embeddings,
                                 kernel=kernel, facts=facts, scoring_type=st)

                xs_np = rs.randint(nb_entities, size=32)
                xp_np = rs.randint(nb_predicates, size=32)
                xo_np = rs.randint(nb_entities, size=32)

                xs_np[0] = 0
                xp_np[0] = 0
                xo_np[0] = 1

                xs_np[1] = 2
                xp_np[1] = 1
                xo_np[1] = 3

                xs = torch.from_numpy(xs_np)
                xp = torch.from_numpy(xp_np)
                xo = torch.from_numpy(xo_np)

                xs_emb = entity_embeddings(xs)
                xp_emb = predicate_embeddings(xp)
                xo_emb = entity_embeddings(xo)

                scores = model.forward(xp_emb, xs_emb, xo_emb)
                inf = model.score(xp_emb, xs_emb, xo_emb)

                assert inf[0] > 0.9
                assert inf[1] > 0.9

                scores_sp, scores_po = scores

                inf = inf.cpu().numpy()
                scores_sp = scores_sp.cpu().numpy()
                scores_po = scores_po.cpu().numpy()

                for i in range(xs.shape[0]):
                    np.testing.assert_allclose(inf[i], scores_sp[i, xo[i]], rtol=1e-2, atol=1e-2)
                    np.testing.assert_allclose(inf[i], scores_po[i, xs[i]], rtol=1e-2, atol=1e-2)


@pytest.mark.light
def test_hoppy_v1():
    nb_entities = 10
    nb_predicates = 5
    embedding_size = 10

    rs = np.random.RandomState(0)

    for _ in range(16):
        for nb_hops in range(6):
            for use_attention in [True, False]:
                with torch.no_grad():
                    entity_embeddings = nn.Embedding(nb_entities, embedding_size * 2, sparse=True)
                    predicate_embeddings = nn.Embedding(nb_predicates, embedding_size * 2, sparse=True)

                    base = ComplEx(entity_embeddings)

                    if use_attention:
                        reformulator = AttentiveReformulator(nb_hops, predicate_embeddings)
                    else:
                        reformulator = LinearReformulator(nb_hops, embedding_size * 2)

                    model = SimpleHoppy(base, entity_embeddings, hops=reformulator)

                    xs = torch.from_numpy(rs.randint(nb_entities, size=32))
                    xp = torch.from_numpy(rs.randint(nb_predicates, size=32))
                    xo = torch.from_numpy(rs.randint(nb_entities, size=32))

                    xs_emb = entity_embeddings(xs)
                    xp_emb = predicate_embeddings(xp)
                    xo_emb = entity_embeddings(xo)

                    scores = model.forward(xp_emb, xs_emb, xo_emb)
                    inf = model.score(xp_emb, xs_emb, xo_emb)

                    scores_sp, scores_po = scores

                    inf = inf.cpu().numpy()
                    scores_sp = scores_sp.cpu().numpy()
                    scores_po = scores_po.cpu().numpy()

                    for i in range(xs.shape[0]):
                        np.testing.assert_allclose(inf[i], scores_sp[i, xo[i]], rtol=1e-3, atol=1e-3)
                        np.testing.assert_allclose(inf[i], scores_po[i, xs[i]], rtol=1e-3, atol=1e-3)


@pytest.mark.light
def test_rhoppy_v1():
    nb_entities = 10
    nb_predicates = 5
    embedding_size = 10

    rs = np.random.RandomState(0)

    for _ in range(8):
        for nb_hops in range(3):
            for depth in range(3):
                for use_attention in [True, False]:
                    with torch.no_grad():
                        entity_embeddings = nn.Embedding(nb_entities, embedding_size * 2, sparse=True)
                        predicate_embeddings = nn.Embedding(nb_predicates, embedding_size * 2, sparse=True)

                        base = ComplEx(entity_embeddings)

                        if use_attention:
                            reformulator = AttentiveReformulator(nb_hops, predicate_embeddings)
                        else:
                            reformulator = LinearReformulator(nb_hops, embedding_size * 2)

                        model = RecursiveHoppy(model=base,
                                               entity_embeddings=entity_embeddings,
                                               hops=reformulator,
                                               depth=depth)

                        xs = torch.from_numpy(rs.randint(nb_entities, size=32))
                        xp = torch.from_numpy(rs.randint(nb_predicates, size=32))
                        xo = torch.from_numpy(rs.randint(nb_entities, size=32))

                        xs_emb = entity_embeddings(xs)
                        xp_emb = predicate_embeddings(xp)
                        xo_emb = entity_embeddings(xo)

                        scores = model.forward(xp_emb, xs_emb, xo_emb)
                        inf = model.score(xp_emb, xs_emb, xo_emb)

                        scores_sp, scores_po = scores

                        inf = inf.cpu().numpy()
                        scores_sp = scores_sp.cpu().numpy()
                        scores_po = scores_po.cpu().numpy()

                        for i in range(xs.shape[0]):
                            np.testing.assert_allclose(inf[i], scores_sp[i, xo[i]], rtol=1e-3, atol=1e-3)
                            np.testing.assert_allclose(inf[i], scores_po[i, xs[i]], rtol=1e-3, atol=1e-3)


@pytest.mark.light
def test_multirhoppy_v1():
    nb_entities = 10
    nb_predicates = 5
    embedding_size = 10

    init_size = 1.0

    rs = np.random.RandomState(0)

    for _ in range(8):
        for nb_hops_lst in [[1], [2], [3], [1, 2], [2, 2], [3, 2], [1, 2, 2], [2, 2, 2], [3, 2, 2]]:
            for depth in range(3):
                for use_attention in [True, False]:
                    with torch.no_grad():
                        entity_embeddings = nn.Embedding(nb_entities, embedding_size * 2, sparse=True)
                        predicate_embeddings = nn.Embedding(nb_predicates, embedding_size * 2, sparse=True)

                        entity_embeddings.weight.data *= init_size
                        predicate_embeddings.weight.data *= init_size

                        base = ComplEx(entity_embeddings)

                        hops_lst = []
                        for i in nb_hops_lst:
                            if use_attention:
                                reformulator = AttentiveReformulator(i, predicate_embeddings)
                            else:
                                reformulator = LinearReformulator(i, embedding_size * 2)
                            hops_lst += [(reformulator, False)]

                        model = Hoppy(model=base,
                                      entity_embeddings=entity_embeddings,
                                      hops_lst=hops_lst,
                                      depth=depth)

                        xs = torch.from_numpy(rs.randint(nb_entities, size=32))
                        xp = torch.from_numpy(rs.randint(nb_predicates, size=32))
                        xo = torch.from_numpy(rs.randint(nb_entities, size=32))

                        xs_emb = entity_embeddings(xs)
                        xp_emb = predicate_embeddings(xp)
                        xo_emb = entity_embeddings(xo)

                        scores = model.forward(xp_emb, xs_emb, xo_emb)
                        inf = model.score(xp_emb, xs_emb, xo_emb)

                        scores_sp, scores_po = scores

                        inf = inf.cpu().numpy()
                        scores_sp = scores_sp.cpu().numpy()
                        scores_po = scores_po.cpu().numpy()

                        for i in range(xs.shape[0]):
                            np.testing.assert_allclose(inf[i], scores_sp[i, xo[i]], rtol=1e-3, atol=1e-3)
                            np.testing.assert_allclose(inf[i], scores_po[i, xs[i]], rtol=1e-3, atol=1e-3)


if __name__ == '__main__':
    pytest.main([__file__])
