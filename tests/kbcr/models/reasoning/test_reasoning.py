# -*- coding: utf-8 -*-

import multiprocessing

import numpy as np

import torch
from torch import nn

from kbcr.kernels import GaussianKernel
from kbcr.models import NeuralKB
from kbcr.models.reasoning import SimpleHoppy, RecursiveHoppy
from kbcr.reformulators import SymbolicReformulator

import pytest


@pytest.mark.light
def test_reasoning_v1():
    torch.set_num_threads(multiprocessing.cpu_count())

    nb_entities = 10
    nb_predicates = 5
    embedding_size = 10

    rs = np.random.RandomState(0)

    triples = [
        ('a', 'p', 'b'),
        ('b', 'q', 'c'),
        ('c', 'r', 'd')
    ]

    entity_to_index = {'a': 0, 'b': 1, 'c': 2, 'd': 3}
    predicate_to_index = {'p': 0, 'q': 1, 'r': 2}

    for st in ['min', 'concat']:
        with torch.no_grad():
            kernel = GaussianKernel()

            entity_embeddings = nn.Embedding(nb_entities, embedding_size * 2, sparse=True)
            predicate_embeddings = nn.Embedding(nb_predicates, embedding_size * 2, sparse=True)

            fact_rel = torch.from_numpy(np.array([predicate_to_index[p] for (_, p, _) in triples]))
            fact_arg1 = torch.from_numpy(np.array([entity_to_index[s] for (s, _, _) in triples]))
            fact_arg2 = torch.from_numpy(np.array([entity_to_index[o] for (_, _, o) in triples]))
            facts = [fact_rel, fact_arg1, fact_arg2]

            model = NeuralKB(entity_embeddings=entity_embeddings, predicate_embeddings=predicate_embeddings,
                             kernel=kernel, facts=facts, scoring_type=st)

            indices = torch.from_numpy(np.array([predicate_to_index['p'], predicate_to_index['q']]))
            reformulator = SymbolicReformulator(predicate_embeddings, indices)
            hoppy = SimpleHoppy(model, entity_embeddings, hops=reformulator)

            xs_np = rs.randint(nb_entities, size=32)
            xp_np = rs.randint(nb_predicates, size=32)
            xo_np = rs.randint(nb_entities, size=32)

            xs_np[0] = 0
            xp_np[0] = 0
            xo_np[0] = 1

            xs_np[1] = 1
            xp_np[1] = 1
            xo_np[1] = 2

            xs_np[2] = 0
            xp_np[2] = 2
            xo_np[2] = 2

            xs = torch.from_numpy(xs_np)
            xp = torch.from_numpy(xp_np)
            xo = torch.from_numpy(xo_np)

            xs_emb = entity_embeddings(xs)
            xp_emb = predicate_embeddings(xp)
            xo_emb = entity_embeddings(xo)

            scores = model.forward(xp_emb, xs_emb, xo_emb)
            inf = model.score(xp_emb, xs_emb, xo_emb)

            scores_h = hoppy.forward(xp_emb, xs_emb, xo_emb)
            inf_h = hoppy.score(xp_emb, xs_emb, xo_emb)

            assert inf[0] > 0.99
            assert inf[1] > 0.99
            assert inf_h[2] > 0.99

            print(inf)
            print(inf_h)

            scores_sp, scores_po = scores
            scores_h_sp, scores_h_po = scores_h

            inf = inf.cpu().numpy()
            scores_sp = scores_sp.cpu().numpy()
            scores_po = scores_po.cpu().numpy()

            inf_h = inf_h.cpu().numpy()
            scores_h_sp = scores_h_sp.cpu().numpy()
            scores_h_po = scores_h_po.cpu().numpy()

            for i in range(xs.shape[0]):
                np.testing.assert_allclose(inf[i], scores_sp[i, xo[i]], rtol=1e-5, atol=1e-5)
                np.testing.assert_allclose(inf[i], scores_po[i, xs[i]], rtol=1e-5, atol=1e-5)

                np.testing.assert_allclose(inf_h[i], scores_h_sp[i, xo[i]], rtol=1e-5, atol=1e-5)
                np.testing.assert_allclose(inf_h[i], scores_h_po[i, xs[i]], rtol=1e-5, atol=1e-5)


@pytest.mark.light
def test_reasoning_v2():
    torch.set_num_threads(multiprocessing.cpu_count())

    nb_entities = 10
    nb_predicates = 5
    embedding_size = 10

    rs = np.random.RandomState(0)

    triples = [
        ('a', 'p', 'b'),
        ('b', 'q', 'c'),
        ('c', 'r', 'd')
    ]

    entity_to_index = {'a': 0, 'b': 1, 'c': 2, 'd': 3}
    predicate_to_index = {'p': 0, 'q': 1, 'r': 2}

    for st in ['min', 'concat']:
        with torch.no_grad():
            kernel = GaussianKernel()

            entity_embeddings = nn.Embedding(nb_entities, embedding_size * 2, sparse=True)
            predicate_embeddings = nn.Embedding(nb_predicates, embedding_size * 2, sparse=True)

            fact_rel = torch.from_numpy(np.array([predicate_to_index[p] for (_, p, _) in triples]))
            fact_arg1 = torch.from_numpy(np.array([entity_to_index[s] for (s, _, _) in triples]))
            fact_arg2 = torch.from_numpy(np.array([entity_to_index[o] for (_, _, o) in triples]))
            facts = [fact_rel, fact_arg1, fact_arg2]

            model = NeuralKB(entity_embeddings=entity_embeddings, predicate_embeddings=predicate_embeddings,
                             kernel=kernel, facts=facts, scoring_type=st)

            indices = torch.from_numpy(np.array([predicate_to_index['p'],
                                                 predicate_to_index['q'],
                                                 predicate_to_index['r']]))
            reformulator = SymbolicReformulator(predicate_embeddings, indices)
            hoppy = SimpleHoppy(model, entity_embeddings, hops=reformulator)

            xs_np = rs.randint(nb_entities, size=32)
            xp_np = rs.randint(nb_predicates, size=32)
            xo_np = rs.randint(nb_entities, size=32)

            xs_np[0] = 0
            xp_np[0] = 0
            xo_np[0] = 1

            xs_np[1] = 1
            xp_np[1] = 1
            xo_np[1] = 2

            xs_np[2] = 0
            xp_np[2] = 2
            xo_np[2] = 3

            xs = torch.from_numpy(xs_np)
            xp = torch.from_numpy(xp_np)
            xo = torch.from_numpy(xo_np)

            xs_emb = entity_embeddings(xs)
            xp_emb = predicate_embeddings(xp)
            xo_emb = entity_embeddings(xo)

            scores = model.forward(xp_emb, xs_emb, xo_emb)
            inf = model.score(xp_emb, xs_emb, xo_emb)

            scores_h = hoppy.forward(xp_emb, xs_emb, xo_emb)
            inf_h = hoppy.score(xp_emb, xs_emb, xo_emb)

            assert inf[0] > 0.99
            assert inf[1] > 0.99
            assert inf_h[2] > 0.99

            print(inf)
            print(inf_h)

            scores_sp, scores_po = scores
            scores_h_sp, scores_h_po = scores_h

            inf = inf.cpu().numpy()
            scores_sp = scores_sp.cpu().numpy()
            scores_po = scores_po.cpu().numpy()

            inf_h = inf_h.cpu().numpy()
            scores_h_sp = scores_h_sp.cpu().numpy()
            scores_h_po = scores_h_po.cpu().numpy()

            for i in range(xs.shape[0]):
                np.testing.assert_allclose(inf[i], scores_sp[i, xo[i]], rtol=1e-5, atol=1e-5)
                np.testing.assert_allclose(inf[i], scores_po[i, xs[i]], rtol=1e-5, atol=1e-5)

                np.testing.assert_allclose(inf_h[i], scores_h_sp[i, xo[i]], rtol=1e-5, atol=1e-5)
                np.testing.assert_allclose(inf_h[i], scores_h_po[i, xs[i]], rtol=1e-5, atol=1e-5)


@pytest.mark.light
def test_reasoning_v3():
    torch.set_num_threads(multiprocessing.cpu_count())

    nb_entities = 10
    nb_predicates = 5
    embedding_size = 10

    rs = np.random.RandomState(0)

    triples = [
        ('a', 'p', 'b'),
        ('b', 'q', 'c'),
        ('c', 'r', 'd'),
        ('d', 's', 'e')
    ]

    entity_to_index = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4}
    predicate_to_index = {'p': 0, 'q': 1, 'r': 2, 's': 3}

    for st in ['min', 'concat']:
        with torch.no_grad():
            kernel = GaussianKernel()

            entity_embeddings = nn.Embedding(nb_entities, embedding_size * 2, sparse=True)
            predicate_embeddings = nn.Embedding(nb_predicates, embedding_size * 2, sparse=True)

            fact_rel = torch.from_numpy(np.array([predicate_to_index[p] for (_, p, _) in triples]))
            fact_arg1 = torch.from_numpy(np.array([entity_to_index[s] for (s, _, _) in triples]))
            fact_arg2 = torch.from_numpy(np.array([entity_to_index[o] for (_, _, o) in triples]))
            facts = [fact_rel, fact_arg1, fact_arg2]

            model = NeuralKB(entity_embeddings=entity_embeddings, predicate_embeddings=predicate_embeddings,
                             kernel=kernel, facts=facts, scoring_type=st)

            indices = torch.from_numpy(np.array([predicate_to_index['p'],
                                                 predicate_to_index['q'],
                                                 predicate_to_index['r'],
                                                 predicate_to_index['s']]))
            reformulator = SymbolicReformulator(predicate_embeddings, indices)
            hoppy = SimpleHoppy(model, entity_embeddings, hops=reformulator)

            xs_np = rs.randint(nb_entities, size=32)
            xp_np = rs.randint(nb_predicates, size=32)
            xo_np = rs.randint(nb_entities, size=32)

            xs_np[0] = 0
            xp_np[0] = 0
            xo_np[0] = 1

            xs_np[1] = 1
            xp_np[1] = 1
            xo_np[1] = 2

            xs_np[2] = 0
            xp_np[2] = 3
            xo_np[2] = 4

            xs = torch.from_numpy(xs_np)
            xp = torch.from_numpy(xp_np)
            xo = torch.from_numpy(xo_np)

            xs_emb = entity_embeddings(xs)
            xp_emb = predicate_embeddings(xp)
            xo_emb = entity_embeddings(xo)

            scores = model.forward(xp_emb, xs_emb, xo_emb)
            inf = model.score(xp_emb, xs_emb, xo_emb)

            scores_h = hoppy.forward(xp_emb, xs_emb, xo_emb)
            inf_h = hoppy.score(xp_emb, xs_emb, xo_emb)

            assert inf[0] > 0.95
            assert inf[1] > 0.95
            assert inf_h[2] > 0.95

            scores_sp, scores_po = scores
            scores_h_sp, scores_h_po = scores_h

            inf = inf.cpu().numpy()
            scores_sp = scores_sp.cpu().numpy()
            scores_po = scores_po.cpu().numpy()

            inf_h = inf_h.cpu().numpy()
            scores_h_sp = scores_h_sp.cpu().numpy()
            scores_h_po = scores_h_po.cpu().numpy()

            for i in range(xs.shape[0]):
                np.testing.assert_allclose(inf[i], scores_sp[i, xo[i]], rtol=1e-5, atol=1e-5)
                np.testing.assert_allclose(inf[i], scores_po[i, xs[i]], rtol=1e-5, atol=1e-5)

                np.testing.assert_allclose(inf_h[i], scores_h_sp[i, xo[i]], rtol=1e-5, atol=1e-5)
                np.testing.assert_allclose(inf_h[i], scores_h_po[i, xs[i]], rtol=1e-5, atol=1e-5)


@pytest.mark.light
def test_reasoning_v4():
    torch.set_num_threads(multiprocessing.cpu_count())

    nb_entities = 10
    nb_predicates = 5
    embedding_size = 10

    rs = np.random.RandomState(0)

    triples = [
        ('a', 'p', 'b'),
        ('b', 'q', 'c'),
        ('c', 'r', 'd'),
        ('d', 's', 'e')
    ]

    entity_to_index = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4}
    predicate_to_index = {'p': 0, 'q': 1, 'r': 2, 's': 3}

    for st in ['min', 'concat']:
        with torch.no_grad():
            kernel = GaussianKernel()

            entity_embeddings = nn.Embedding(nb_entities, embedding_size * 2, sparse=True)
            predicate_embeddings = nn.Embedding(nb_predicates, embedding_size * 2, sparse=True)

            fact_rel = torch.from_numpy(np.array([predicate_to_index[p] for (_, p, _) in triples]))
            fact_arg1 = torch.from_numpy(np.array([entity_to_index[s] for (s, _, _) in triples]))
            fact_arg2 = torch.from_numpy(np.array([entity_to_index[o] for (_, _, o) in triples]))
            facts = [fact_rel, fact_arg1, fact_arg2]

            model = NeuralKB(entity_embeddings=entity_embeddings, predicate_embeddings=predicate_embeddings,
                             kernel=kernel, facts=facts, scoring_type=st)

            indices = torch.from_numpy(np.array([predicate_to_index['p'],
                                                 predicate_to_index['q'],
                                                 predicate_to_index['r'],
                                                 predicate_to_index['s']]))
            reformulator = SymbolicReformulator(predicate_embeddings, indices)
            rhoppy = RecursiveHoppy(model, entity_embeddings, hops=reformulator, depth=0)

            xs_np = rs.randint(nb_entities, size=32)
            xp_np = rs.randint(nb_predicates, size=32)
            xo_np = rs.randint(nb_entities, size=32)

            xs_np[0] = 0
            xp_np[0] = 0
            xo_np[0] = 1

            xs_np[1] = 1
            xp_np[1] = 1
            xo_np[1] = 2

            xs_np[2] = 0
            xp_np[2] = 3
            xo_np[2] = 4

            xs = torch.from_numpy(xs_np)
            xp = torch.from_numpy(xp_np)
            xo = torch.from_numpy(xo_np)

            xs_emb = entity_embeddings(xs)
            xp_emb = predicate_embeddings(xp)
            xo_emb = entity_embeddings(xo)

            scores = model.forward(xp_emb, xs_emb, xo_emb)
            inf = model.score(xp_emb, xs_emb, xo_emb)

            scores_h = rhoppy.forward(xp_emb, xs_emb, xo_emb)
            inf_h = rhoppy.score(xp_emb, xs_emb, xo_emb)

            print(inf)
            print(inf_h)

            assert inf[0] > 0.95
            assert inf[1] > 0.95

            scores_sp, scores_po = scores
            scores_h_sp, scores_h_po = scores_h

            inf = inf.cpu().numpy()
            scores_sp = scores_sp.cpu().numpy()
            scores_po = scores_po.cpu().numpy()

            inf_h = inf_h.cpu().numpy()
            scores_h_sp = scores_h_sp.cpu().numpy()
            scores_h_po = scores_h_po.cpu().numpy()

            np.testing.assert_allclose(inf, inf_h)
            np.testing.assert_allclose(scores_sp, scores_h_sp)
            np.testing.assert_allclose(scores_po, scores_h_po)

            for i in range(xs.shape[0]):
                np.testing.assert_allclose(inf[i], scores_sp[i, xo[i]], rtol=1e-5, atol=1e-5)
                np.testing.assert_allclose(inf[i], scores_po[i, xs[i]], rtol=1e-5, atol=1e-5)

                np.testing.assert_allclose(inf_h[i], scores_h_sp[i, xo[i]], rtol=1e-5, atol=1e-5)
                np.testing.assert_allclose(inf_h[i], scores_h_po[i, xs[i]], rtol=1e-5, atol=1e-5)


@pytest.mark.light
def test_reasoning_v5():
    torch.set_num_threads(multiprocessing.cpu_count())

    nb_entities = 10
    nb_predicates = 5
    embedding_size = 10

    rs = np.random.RandomState(0)

    triples = [
        ('a', 'p', 'b'),
        ('b', 'q', 'c'),
        ('c', 'r', 'd'),
        ('d', 's', 'e')
    ]

    entity_to_index = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4}
    predicate_to_index = {'p': 0, 'q': 1, 'r': 2, 's': 3}

    for st in ['min', 'concat']:
        with torch.no_grad():
            kernel = GaussianKernel()

            entity_embeddings = nn.Embedding(nb_entities, embedding_size * 2, sparse=True)
            predicate_embeddings = nn.Embedding(nb_predicates, embedding_size * 2, sparse=True)

            fact_rel = torch.from_numpy(np.array([predicate_to_index[p] for (_, p, _) in triples]))
            fact_arg1 = torch.from_numpy(np.array([entity_to_index[s] for (s, _, _) in triples]))
            fact_arg2 = torch.from_numpy(np.array([entity_to_index[o] for (_, _, o) in triples]))
            facts = [fact_rel, fact_arg1, fact_arg2]

            model = NeuralKB(entity_embeddings=entity_embeddings, predicate_embeddings=predicate_embeddings,
                             kernel=kernel, facts=facts, scoring_type=st)

            indices = torch.from_numpy(np.array([predicate_to_index['p'],
                                                 predicate_to_index['q'],
                                                 predicate_to_index['r'],
                                                 predicate_to_index['s']]))
            reformulator = SymbolicReformulator(predicate_embeddings, indices)
            hoppy = SimpleHoppy(model, entity_embeddings, hops=reformulator)
            rhoppy = RecursiveHoppy(model, entity_embeddings, hops=reformulator, depth=1)

            xs_np = rs.randint(nb_entities, size=32)
            xp_np = rs.randint(nb_predicates, size=32)
            xo_np = rs.randint(nb_entities, size=32)

            xs_np[0] = 0
            xp_np[0] = 0
            xo_np[0] = 1

            xs_np[1] = 1
            xp_np[1] = 1
            xo_np[1] = 2

            xs_np[2] = 0
            xp_np[2] = 3
            xo_np[2] = 4

            xs = torch.from_numpy(xs_np)
            xp = torch.from_numpy(xp_np)
            xo = torch.from_numpy(xo_np)

            xs_emb = entity_embeddings(xs)
            xp_emb = predicate_embeddings(xp)
            xo_emb = entity_embeddings(xo)

            scores = hoppy.forward(xp_emb, xs_emb, xo_emb)
            inf = hoppy.score(xp_emb, xs_emb, xo_emb)

            scores_h = rhoppy.depth_r_forward(xp_emb, xs_emb, xo_emb, depth=1)
            inf_h = rhoppy.depth_r_score(xp_emb, xs_emb, xo_emb, depth=1)

            print(inf)
            print(inf_h)

            assert inf[2] > 0.95

            scores_sp, scores_po = scores
            scores_h_sp, scores_h_po = scores_h

            inf = inf.cpu().numpy()
            scores_sp = scores_sp.cpu().numpy()
            scores_po = scores_po.cpu().numpy()

            inf_h = inf_h.cpu().numpy()
            scores_h_sp = scores_h_sp.cpu().numpy()
            scores_h_po = scores_h_po.cpu().numpy()

            np.testing.assert_allclose(inf, inf_h)
            np.testing.assert_allclose(scores_sp, scores_h_sp)
            np.testing.assert_allclose(scores_po, scores_h_po)

            for i in range(xs.shape[0]):
                np.testing.assert_allclose(inf[i], scores_sp[i, xo[i]], rtol=1e-5, atol=1e-5)
                np.testing.assert_allclose(inf[i], scores_po[i, xs[i]], rtol=1e-5, atol=1e-5)

                np.testing.assert_allclose(inf_h[i], scores_h_sp[i, xo[i]], rtol=1e-5, atol=1e-5)
                np.testing.assert_allclose(inf_h[i], scores_h_po[i, xs[i]], rtol=1e-5, atol=1e-5)


@pytest.mark.light
def test_reasoning_v6():
    torch.set_num_threads(multiprocessing.cpu_count())

    embedding_size = 50

    torch.manual_seed(0)
    rs = np.random.RandomState(0)

    triples = [
        ('a', 'p', 'b'),
        ('b', 'q', 'c'),
        ('c', 'p', 'd'),
        ('d', 'q', 'e'),
        ('e', 'p', 'f'),
        ('f', 'q', 'g'),
        ('g', 'p', 'h'),
        ('h', 'q', 'i'),
        ('i', 'p', 'l'),
        ('l', 'q', 'm'),
        ('m', 'p', 'n'),
        ('n', 'q', 'o'),
        ('o', 'p', 'p'),
        ('p', 'q', 'q'),
        ('q', 'p', 'r'),
        ('r', 'q', 's'),
        ('s', 'p', 't'),
        ('t', 'q', 'u'),
        ('u', 'p', 'v'),
        ('v', 'q', 'w'),

        ('x', 'r', 'y'),
        ('x', 's', 'y')
    ]

    entity_lst = sorted({e for (e, _, _) in triples} | {e for (_, _, e) in triples})
    predicate_lst = sorted({p for (_, p, _) in triples})

    nb_entities = len(entity_lst)
    nb_predicates = len(predicate_lst)

    entity_to_index = {e: i for i, e in enumerate(entity_lst)}
    predicate_to_index = {p: i for i, p in enumerate(predicate_lst)}

    for st in ['min', 'concat']:
        with torch.no_grad():
            kernel = GaussianKernel()

            entity_embeddings = nn.Embedding(nb_entities, embedding_size * 2, sparse=True)
            predicate_embeddings = nn.Embedding(nb_predicates, embedding_size * 2, sparse=True)

            fact_rel = torch.from_numpy(np.array([predicate_to_index[p] for (_, p, _) in triples]))
            fact_arg1 = torch.from_numpy(np.array([entity_to_index[s] for (s, _, _) in triples]))
            fact_arg2 = torch.from_numpy(np.array([entity_to_index[o] for (_, _, o) in triples]))
            facts = [fact_rel, fact_arg1, fact_arg2]

            model = NeuralKB(entity_embeddings=entity_embeddings, predicate_embeddings=predicate_embeddings,
                             kernel=kernel, facts=facts, scoring_type=st)

            indices = torch.from_numpy(np.array([predicate_to_index['p'],
                                                 predicate_to_index['q']]))
            reformulator = SymbolicReformulator(predicate_embeddings, indices)

            k = 5

            rhoppy0 = RecursiveHoppy(model, entity_embeddings, hops=reformulator, depth=0, k=k)
            rhoppy1 = RecursiveHoppy(model, entity_embeddings, hops=reformulator, depth=1, k=k)
            rhoppy2 = RecursiveHoppy(model, entity_embeddings, hops=reformulator, depth=2, k=k)
            rhoppy3 = RecursiveHoppy(model, entity_embeddings, hops=reformulator, depth=3, k=k)
            rhoppy4 = RecursiveHoppy(model, entity_embeddings, hops=reformulator, depth=4, k=k)

            xs_np = rs.randint(nb_entities, size=12)
            xp_np = rs.randint(nb_predicates, size=12)
            xo_np = rs.randint(nb_entities, size=12)

            xs_np[0] = entity_to_index['a']
            xp_np[0] = predicate_to_index['r']
            xo_np[0] = entity_to_index['c']

            xs_np[1] = entity_to_index['a']
            xp_np[1] = predicate_to_index['r']
            xo_np[1] = entity_to_index['e']

            xs_np[2] = entity_to_index['a']
            xp_np[2] = predicate_to_index['r']
            xo_np[2] = entity_to_index['g']

            xs_np[3] = entity_to_index['a']
            xp_np[3] = predicate_to_index['r']
            xo_np[3] = entity_to_index['i']

            xs_np[4] = entity_to_index['a']
            xp_np[4] = predicate_to_index['r']
            xo_np[4] = entity_to_index['m']

            xs_np[5] = entity_to_index['a']
            xp_np[5] = predicate_to_index['r']
            xo_np[5] = entity_to_index['o']

            xs_np[6] = entity_to_index['a']
            xp_np[6] = predicate_to_index['r']
            xo_np[6] = entity_to_index['q']

            xs_np[7] = entity_to_index['a']
            xp_np[7] = predicate_to_index['r']
            xo_np[7] = entity_to_index['s']

            xs_np[8] = entity_to_index['a']
            xp_np[8] = predicate_to_index['r']
            xo_np[8] = entity_to_index['u']

            # xs_np[9] = entity_to_index['a']
            # xp_np[9] = predicate_to_index['r']
            # xo_np[9] = entity_to_index['w']

            xs = torch.from_numpy(xs_np)
            xp = torch.from_numpy(xp_np)
            xo = torch.from_numpy(xo_np)

            xs_emb = entity_embeddings(xs)
            xp_emb = predicate_embeddings(xp)
            xo_emb = entity_embeddings(xo)

            scores0 = rhoppy0.forward(xp_emb, xs_emb, xo_emb)
            inf0 = rhoppy0.score(xp_emb, xs_emb, xo_emb)

            for i in range(xs.shape[0]):
                scores_sp, scores_po = scores0
                inf_np = inf0.cpu().numpy()

                scores_sp_np = scores_sp.cpu().numpy()
                scores_po_np = scores_po.cpu().numpy()

                np.testing.assert_allclose(inf_np[i], scores_sp_np[i, xo[i]], rtol=1e-5, atol=1e-5)
                np.testing.assert_allclose(inf_np[i], scores_po_np[i, xs[i]], rtol=1e-5, atol=1e-5)

            scores1 = rhoppy1.forward(xp_emb, xs_emb, xo_emb)
            inf1 = rhoppy1.score(xp_emb, xs_emb, xo_emb)

            for i in range(xs.shape[0]):
                scores_sp, scores_po = scores1
                inf_np = inf1.cpu().numpy()

                scores_sp_np = scores_sp.cpu().numpy()
                scores_po_np = scores_po.cpu().numpy()

                np.testing.assert_allclose(inf_np[i], scores_sp_np[i, xo[i]], rtol=1e-5, atol=1e-5)
                np.testing.assert_allclose(inf_np[i], scores_po_np[i, xs[i]], rtol=1e-5, atol=1e-5)

            scores2 = rhoppy2.forward(xp_emb, xs_emb, xo_emb)
            inf2 = rhoppy2.score(xp_emb, xs_emb, xo_emb)

            for i in range(xs.shape[0]):
                scores_sp, scores_po = scores2
                inf_np = inf2.cpu().numpy()

                scores_sp_np = scores_sp.cpu().numpy()
                scores_po_np = scores_po.cpu().numpy()

                np.testing.assert_allclose(inf_np[i], scores_sp_np[i, xo[i]], rtol=1e-1, atol=1e-1)
                np.testing.assert_allclose(inf_np[i], scores_po_np[i, xs[i]], rtol=1e-1, atol=1e-1)

            scores3 = rhoppy3.forward(xp_emb, xs_emb, xo_emb)
            inf3 = rhoppy3.score(xp_emb, xs_emb, xo_emb)

            for i in range(xs.shape[0]):
                scores_sp, scores_po = scores3
                inf_np = inf3.cpu().numpy()

                scores_sp_np = scores_sp.cpu().numpy()
                scores_po_np = scores_po.cpu().numpy()

                np.testing.assert_allclose(inf_np[i], scores_sp_np[i, xo[i]], rtol=1e-1, atol=1e-1)
                np.testing.assert_allclose(inf_np[i], scores_po_np[i, xs[i]], rtol=1e-1, atol=1e-1)

            scores4 = rhoppy4.forward(xp_emb, xs_emb, xo_emb)
            inf4 = rhoppy4.score(xp_emb, xs_emb, xo_emb)

            for i in range(xs.shape[0]):
                scores_sp, scores_po = scores4
                inf_np = inf4.cpu().numpy()

                scores_sp_np = scores_sp.cpu().numpy()
                scores_po_np = scores_po.cpu().numpy()

                np.testing.assert_allclose(inf_np[i], scores_sp_np[i, xo[i]], rtol=1e-1, atol=1e-1)
                np.testing.assert_allclose(inf_np[i], scores_po_np[i, xs[i]], rtol=1e-1, atol=1e-1)

            print(inf0)
            print(inf1)
            print(inf2)
            print(inf3)
            print(inf4)

            inf0_np = inf0.cpu().numpy()
            inf1_np = inf1.cpu().numpy()
            inf2_np = inf2.cpu().numpy()
            inf3_np = inf3.cpu().numpy()
            inf4_np = inf4.cpu().numpy()

            np.testing.assert_allclose(inf0_np, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], rtol=1e-1, atol=1e-1)
            np.testing.assert_allclose(inf1_np, [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], rtol=1e-1, atol=1e-1)
            np.testing.assert_allclose(inf2_np, [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], rtol=1e-1, atol=1e-1)
            np.testing.assert_allclose(inf3_np, [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0], rtol=1e-1, atol=1e-1)
            np.testing.assert_allclose(inf4_np, [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0], rtol=1e-1, atol=1e-1)


if __name__ == '__main__':
    pytest.main([__file__])
    # test_reasoning_v6()
