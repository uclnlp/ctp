# -*- coding: utf-8 -*-

import numpy as np

import multiprocessing
from itertools import cycle, islice

import torch
from torch import nn, optim, Tensor

from kbcr.kernels import GaussianKernel
from kbcr.clutrr.models import NeuralKB, Hoppy
from kbcr.reformulators import AttentiveReformulator
from kbcr.reformulators import SymbolicReformulator

from kbcr.util import make_batches

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
def test_clutrr_v1():
    embedding_size = 50

    triples, hops = [], []

    for i in range(16):
        triples += [(f'a{i}', 'p', f'b{i}'), (f'b{i}', 'q', f'c{i}')]
        hops += [(f'a{i}', 'r', f'c{i}')]

    entity_lst = sorted({e for (e, _, _) in triples + hops} | {e for (e, _, e) in triples + hops})
    predicate_lst = sorted({p for (_, p, _) in triples + hops})

    nb_entities, nb_predicates = len(entity_lst), len(predicate_lst)

    entity_to_index = {e: i for i, e in enumerate(entity_lst)}
    predicate_to_index = {p: i for i, p in enumerate(predicate_lst)}

    kernel = GaussianKernel()

    entity_embeddings = nn.Embedding(nb_entities, embedding_size, sparse=True)
    predicate_embeddings = nn.Embedding(nb_predicates, embedding_size, sparse=True)

    for scoring_type in ['concat']:  # ['min', 'concat']:
        model = NeuralKB(kernel=kernel, scoring_type=scoring_type)

        for s in entity_lst:
            for p in predicate_lst:
                for o in entity_lst:
                    xs_np = np.array([entity_to_index[s]])
                    xp_np = np.array([predicate_to_index[p]])
                    xo_np = np.array([entity_to_index[o]])

                    with torch.no_grad():
                        xs = torch.from_numpy(xs_np)
                        xp = torch.from_numpy(xp_np)
                        xo = torch.from_numpy(xo_np)

                        xs_emb = entity_embeddings(xs)
                        xp_emb = predicate_embeddings(xp)
                        xo_emb = entity_embeddings(xo)

                        rel_emb = encode_relation(facts=triples, relation_embeddings=predicate_embeddings,
                                                  relation_to_idx=predicate_to_index)
                        arg1_emb, arg2_emb = encode_arguments(facts=triples, entity_embeddings=entity_embeddings,
                                                              entity_to_idx=entity_to_index)

                        facts = [rel_emb, arg1_emb, arg2_emb]

                        inf = model.score(xp_emb, xs_emb, xo_emb, facts=facts)
                        inf_np = inf.cpu().numpy()

                        assert inf_np[0] > 0.95 if (s, p, o) in triples else inf_np[0] < 0.01


@pytest.mark.light
def test_clutrr_v2():
    embedding_size = 20

    triples, hops = [], []
    xxx = []

    for i in range(16):
        triples += [(f'a{i}', 'p', f'b{i}'), (f'b{i}', 'q', f'c{i}')]
        hops += [(f'a{i}', 'r', f'c{i}')]
        xxx += [(f'a{i}', 'p', f'c{i}'), (f'a{i}', 'q', f'c{i}'), (f'a{i}', 'r', f'c{i}')]

    entity_lst = sorted({s for (s, _, _) in triples + hops} | {o for (_, _, o) in triples + hops})
    predicate_lst = sorted({p for (_, p, _) in triples + hops})

    nb_entities, nb_predicates = len(entity_lst), len(predicate_lst)

    entity_to_index = {e: i for i, e in enumerate(entity_lst)}
    predicate_to_index = {p: i for i, p in enumerate(predicate_lst)}

    kernel = GaussianKernel()

    entity_embeddings = nn.Embedding(nb_entities, embedding_size, sparse=True)
    predicate_embeddings = nn.Embedding(nb_predicates, embedding_size, sparse=True)

    for scoring_type in ['concat']:  # ['min', 'concat']:
        model = NeuralKB(kernel=kernel, scoring_type=scoring_type)

        indices = torch.from_numpy(np.array([predicate_to_index['p'], predicate_to_index['q']]))
        _hops = SymbolicReformulator(predicate_embeddings, indices)
        hoppy = Hoppy(model, hops_lst=[(_hops, False)], depth=1)

        for s in entity_lst:
            for p in predicate_lst:
                for o in entity_lst:
                    xs_np = np.array([entity_to_index[s]])
                    xp_np = np.array([predicate_to_index[p]])
                    xo_np = np.array([entity_to_index[o]])

                    with torch.no_grad():
                        xs = torch.from_numpy(xs_np)
                        xp = torch.from_numpy(xp_np)
                        xo = torch.from_numpy(xo_np)

                        xs_emb = entity_embeddings(xs)
                        xp_emb = predicate_embeddings(xp)
                        xo_emb = entity_embeddings(xo)

                        rel_emb = encode_relation(facts=triples, relation_embeddings=predicate_embeddings,
                                                  relation_to_idx=predicate_to_index)
                        arg1_emb, arg2_emb = encode_arguments(facts=triples, entity_embeddings=entity_embeddings,
                                                              entity_to_idx=entity_to_index)

                        facts = [rel_emb, arg1_emb, arg2_emb]

                        inf = hoppy.score(xp_emb, xs_emb, xo_emb, facts=facts,
                                          entity_embeddings=entity_embeddings.weight)
                        inf_np = inf.cpu().numpy()

                        print(s, p, o, inf_np)
                        assert inf_np[0] > 0.9 if (s, p, o) in (triples + xxx) else inf_np[0] < 0.1


@pytest.mark.light
def test_clutrr_v3():
    embedding_size = 20
    batch_size = 8

    torch.manual_seed(0)

    triples, hops = [], []

    for i in range(32):
        triples += [(f'a{i}', 'p', f'b{i}'), (f'b{i}', 'q', f'c{i}')]
        hops += [(f'a{i}', 'r', f'c{i}')]

    entity_lst = sorted({s for (s, _, _) in triples + hops} | {o for (_, _, o) in triples + hops})
    predicate_lst = sorted({p for (_, p, _) in triples + hops})

    nb_entities, nb_predicates = len(entity_lst), len(predicate_lst)

    entity_to_index = {e: i for i, e in enumerate(entity_lst)}
    predicate_to_index = {p: i for i, p in enumerate(predicate_lst)}

    kernel = GaussianKernel(slope=None)

    entity_embeddings = nn.Embedding(nb_entities, embedding_size, sparse=True)
    predicate_embeddings = nn.Embedding(nb_predicates, embedding_size, sparse=True)

    # _hops = LinearReformulator(2, embedding_size)
    _hops = AttentiveReformulator(2, predicate_embeddings)

    model = NeuralKB(kernel=kernel, scoring_type='concat')
    hoppy = Hoppy(model, hops_lst=[(_hops, False)], depth=1)

    params = [p for p in hoppy.parameters()
              if not torch.equal(p, entity_embeddings.weight)
              and not torch.equal(p, predicate_embeddings.weight)]

    for tensor in params:
        print(f'\t{tensor.size()}\t{tensor.device}')

    loss_function = nn.BCELoss()

    optimizer = optim.Adagrad(params, lr=0.1)

    hops_data = []
    for i in range(64):
        hops_data += hops

    batches = make_batches(len(hops_data), batch_size)

    rs = np.random.RandomState()

    c, d = 0.0, 0.0
    p_emb = predicate_embeddings(torch.from_numpy(np.array([predicate_to_index['p']])))
    q_emb = predicate_embeddings(torch.from_numpy(np.array([predicate_to_index['q']])))

    for batch_start, batch_end in batches:
        hops_batch = hops_data[batch_start:batch_end]

        s_lst = [s for (s, _, _) in hops_batch]
        p_lst = [p for (_, p, _) in hops_batch]
        o_lst = [o for (_, _, o) in hops_batch]

        nb_positives = len(s_lst)
        nb_negatives = nb_positives * 3

        s_n_lst = rs.permutation(nb_entities)[:nb_negatives].tolist()
        nb_negatives = len(s_n_lst)
        o_n_lst = rs.permutation(nb_entities)[:nb_negatives].tolist()
        p_n_lst = list(islice(cycle(p_lst), nb_negatives))

        xs_np = np.array([entity_to_index[s] for s in s_lst] + s_n_lst)
        xp_np = np.array([predicate_to_index[p] for p in p_lst + p_n_lst])
        xo_np = np.array([entity_to_index[o] for o in o_lst] + o_n_lst)

        xs_emb = entity_embeddings(torch.from_numpy(xs_np))
        xp_emb = predicate_embeddings(torch.from_numpy(xp_np))
        xo_emb = entity_embeddings(torch.from_numpy(xo_np))

        rel_emb = encode_relation(facts=triples, relation_embeddings=predicate_embeddings,
                                  relation_to_idx=predicate_to_index)
        arg1_emb, arg2_emb = encode_arguments(facts=triples, entity_embeddings=entity_embeddings,
                                              entity_to_idx=entity_to_index)

        facts = [rel_emb, arg1_emb, arg2_emb]

        scores = hoppy.score(xp_emb, xs_emb, xo_emb, facts=facts, entity_embeddings=entity_embeddings.weight)

        labels_np = np.zeros(xs_np.shape[0])
        labels_np[:nb_positives] = 1
        labels = torch.from_numpy(labels_np).float()

        # for s, p, o, l in zip(xs_np, xp_np, xo_np, labels):
        #     print(s, p, o, l)

        loss = loss_function(scores, labels)

        hop_1_emb = hoppy.hops_lst[0][0].hops_lst[0](xp_emb)
        hop_2_emb = hoppy.hops_lst[0][0].hops_lst[1](xp_emb)

        c = kernel.pairwise(p_emb, hop_1_emb).mean().cpu().detach().numpy()
        d = kernel.pairwise(q_emb, hop_2_emb).mean().cpu().detach().numpy()

        print(c, d)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    assert c > 0.95 and d > 0.95


@pytest.mark.light
def test_clutrr_v4():
    embedding_size = 50

    rs = np.random.RandomState(0)

    for _ in range(32):
        with torch.no_grad():
            triples = [
                ('a', 'p', 'b'),
                ('c', 'q', 'd'),
                ('e', 'q', 'f'),
                ('g', 'q', 'h'),
                ('i', 'q', 'l'),
                ('m', 'q', 'n'),
                ('o', 'q', 'p'),
                ('q', 'q', 'r'),
                ('s', 'q', 't'),
                ('u', 'q', 'v')
            ]

            entity_lst = sorted({s for (s, _, _) in triples} | {o for (_, _, o) in triples})
            predicate_lst = sorted({p for (_, p, _) in triples})

            nb_entities, nb_predicates = len(entity_lst), len(predicate_lst)

            entity_to_index = {e: i for i, e in enumerate(entity_lst)}
            predicate_to_index = {p: i for i, p in enumerate(predicate_lst)}

            kernel = GaussianKernel()

            entity_embeddings = nn.Embedding(nb_entities, embedding_size * 2, sparse=True)
            predicate_embeddings = nn.Embedding(nb_predicates, embedding_size * 2, sparse=True)

            rel_emb = encode_relation(facts=triples, relation_embeddings=predicate_embeddings,
                                      relation_to_idx=predicate_to_index)
            arg1_emb, arg2_emb = encode_arguments(facts=triples, entity_embeddings=entity_embeddings,
                                                  entity_to_idx=entity_to_index)

            facts = [rel_emb, arg1_emb, arg2_emb]

            model = NeuralKB(kernel=kernel)

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

            print('xp_emb', xp_emb.shape)

            scores_sp, scores_po = model.forward(xp_emb, xs_emb, xo_emb, facts=facts, entity_embeddings=entity_embeddings.weight)
            inf = model.score(xp_emb, xs_emb, xo_emb, facts=facts)

            assert inf[0] > 0.9
            assert inf[1] > 0.9

            inf = inf.cpu().numpy()
            scores_sp = scores_sp.cpu().numpy()
            scores_po = scores_po.cpu().numpy()

            print('AAA', inf)
            print('BBB', scores_sp)


@pytest.mark.light
def test_clutrr_v5():
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

    with torch.no_grad():
        kernel = GaussianKernel()

        entity_embeddings = nn.Embedding(nb_entities, embedding_size * 2, sparse=True)
        predicate_embeddings = nn.Embedding(nb_predicates, embedding_size * 2, sparse=True)

        rel_emb = encode_relation(facts=triples,
                                  relation_embeddings=predicate_embeddings,
                                  relation_to_idx=predicate_to_index)

        arg1_emb, arg2_emb = encode_arguments(facts=triples,
                                              entity_embeddings=entity_embeddings,
                                              entity_to_idx=entity_to_index)

        facts = [rel_emb, arg1_emb, arg2_emb]

        model = NeuralKB(kernel=kernel)

        indices = torch.from_numpy(np.array([predicate_to_index['p'], predicate_to_index['q']]))
        reformulator = SymbolicReformulator(predicate_embeddings, indices)

        hoppy0 = Hoppy(model, hops_lst=[(reformulator, False), (reformulator, False)], depth=0)
        hoppy1 = Hoppy(model, hops_lst=[(reformulator, False), (reformulator, False)], depth=1)
        hoppy2 = Hoppy(model, hops_lst=[(reformulator, False), (reformulator, False)], depth=2)
        hoppy3 = Hoppy(model, hops_lst=[(reformulator, False), (reformulator, False)], depth=3)
        hoppy4 = Hoppy(model, hops_lst=[(reformulator, False), (reformulator, False)], depth=4)

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

        # scores0 = hoppy0.forward(xp_emb, xs_emb, xo_emb, facts=facts, entity_embeddings=entity_embeddings)
        inf0 = hoppy0.score(xp_emb, xs_emb, xo_emb, facts=facts, entity_embeddings=entity_embeddings.weight)

        # scores1 = hoppy1.forward(xp_emb, xs_emb, xo_emb, facts=facts, entity_embeddings=entity_embeddings)
        inf1 = hoppy1.score(xp_emb, xs_emb, xo_emb, facts=facts, entity_embeddings=entity_embeddings.weight)

        # scores2 = hoppy2.forward(xp_emb, xs_emb, xo_emb, facts=facts, entity_embeddings=entity_embeddings)
        inf2 = hoppy2.score(xp_emb, xs_emb, xo_emb, facts=facts, entity_embeddings=entity_embeddings.weight)

        # scores3 = hoppy3.forward(xp_emb, xs_emb, xo_emb, facts=facts, entity_embeddings=entity_embeddings)
        inf3 = hoppy3.score(xp_emb, xs_emb, xo_emb, facts=facts, entity_embeddings=entity_embeddings.weight)

        # scores4 = hoppy4.forward(xp_emb, xs_emb, xo_emb, facts=facts, entity_embeddings=entity_embeddings)
        inf4 = hoppy4.score(xp_emb, xs_emb, xo_emb, facts=facts, entity_embeddings=entity_embeddings.weight)

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


@pytest.mark.light
def test_clutrr_v6():
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

    for st in ['concat']:
        with torch.no_grad():
            kernel = GaussianKernel()

            entity_embeddings = nn.Embedding(nb_entities, embedding_size * 2, sparse=True)
            predicate_embeddings = nn.Embedding(nb_predicates, embedding_size * 2, sparse=True)

            rel_emb = encode_relation(facts=triples,
                                      relation_embeddings=predicate_embeddings,
                                      relation_to_idx=predicate_to_index)

            arg1_emb, arg2_emb = encode_arguments(facts=triples,
                                                  entity_embeddings=entity_embeddings,
                                                  entity_to_idx=entity_to_index)

            facts = [rel_emb, arg1_emb, arg2_emb]

            k = 5

            model = NeuralKB(kernel=kernel)

            indices = torch.from_numpy(np.array([predicate_to_index['p'], predicate_to_index['q']]))
            reformulator = SymbolicReformulator(predicate_embeddings, indices)

            hoppy0 = Hoppy(model, hops_lst=[(reformulator, False)], depth=0)
            hoppy1 = Hoppy(model, hops_lst=[(reformulator, False)], depth=1)
            hoppy2 = Hoppy(model, hops_lst=[(reformulator, False)], depth=2)
            hoppy3 = Hoppy(model, hops_lst=[(reformulator, False)], depth=3)
            hoppy4 = Hoppy(model, hops_lst=[(reformulator, False)], depth=4)

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

            xs = torch.from_numpy(xs_np)
            xp = torch.from_numpy(xp_np)
            xo = torch.from_numpy(xo_np)

            xs_emb = entity_embeddings(xs)
            xp_emb = predicate_embeddings(xp)
            xo_emb = entity_embeddings(xo)

            # res0 = hoppy0.forward(xp_emb, xs_emb, xo_emb, facts=facts, entity_embeddings=entity_embeddings)
            inf0 = hoppy0.score(xp_emb, xs_emb, xo_emb, facts=facts, entity_embeddings=entity_embeddings.weight)

            # (scores0_sp, subs0_sp), (scores0_po, subs0_po) = res0

            # res1 = hoppy1.forward(xp_emb, xs_emb, xo_emb, facts=facts, entity_embeddings=entity_embeddings)
            inf1 = hoppy1.score(xp_emb, xs_emb, xo_emb, facts=facts, entity_embeddings=entity_embeddings.weight)

            # (scores1_sp, subs1_sp), (scores1_po, subs1_po) = res1

            # res2 = hoppy2.forward(xp_emb, xs_emb, xo_emb, facts=facts, entity_embeddings=entity_embeddings)
            inf2 = hoppy2.score(xp_emb, xs_emb, xo_emb, facts=facts, entity_embeddings=entity_embeddings.weight)

            # (scores2_sp, subs2_sp), (scores2_po, subs2_po) = res2

            # res3 = hoppy3.forward(xp_emb, xs_emb, xo_emb, facts=facts, entity_embeddings=entity_embeddings)
            inf3 = hoppy3.score(xp_emb, xs_emb, xo_emb, facts=facts, entity_embeddings=entity_embeddings.weight)

            # (scores3_sp, subs3_sp), (scores3_po, subs3_po) = res3

            # scores4 = hoppy4.forward(xp_emb, xs_emb, xo_emb, facts=facts, entity_embeddings=entity_embeddings)
            inf4 = hoppy4.score(xp_emb, xs_emb, xo_emb, facts=facts, entity_embeddings=entity_embeddings.weight)

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


@pytest.mark.light
def test_clutrr_v7():
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

    with torch.no_grad():
        kernel = GaussianKernel()

        entity_embeddings = nn.Embedding(nb_entities, embedding_size * 2, sparse=True)
        predicate_embeddings = nn.Embedding(nb_predicates, embedding_size * 2, sparse=True)

        rel_emb = encode_relation(facts=triples,
                                  relation_embeddings=predicate_embeddings,
                                  relation_to_idx=predicate_to_index)

        arg1_emb, arg2_emb = encode_arguments(facts=triples,
                                              entity_embeddings=entity_embeddings,
                                              entity_to_idx=entity_to_index)

        facts = [rel_emb, arg1_emb, arg2_emb]

        k = 5

        model = NeuralKB(kernel=kernel)

        indices = torch.from_numpy(np.array([predicate_to_index['p'], predicate_to_index['q']]))
        reformulator = SymbolicReformulator(predicate_embeddings, indices)

        hoppy0 = Hoppy(model, hops_lst=[(reformulator, False)], depth=0)
        hoppy1 = Hoppy(model, hops_lst=[(reformulator, False)], depth=1)
        hoppy2 = Hoppy(model, hops_lst=[(reformulator, False)], depth=2)
        hoppy3 = Hoppy(model, hops_lst=[(reformulator, False)], depth=3)
        hoppy4 = Hoppy(model, hops_lst=[(reformulator, False)], depth=4)

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

        xs = torch.from_numpy(xs_np)
        xp = torch.from_numpy(xp_np)
        xo = torch.from_numpy(xo_np)

        xs_emb = entity_embeddings(xs)
        xp_emb = predicate_embeddings(xp)
        xo_emb = entity_embeddings(xo)

        # res0 = hoppy0.forward(xp_emb, xs_emb, xo_emb, facts=facts, entity_embeddings=entity_embeddings)
        inf0 = hoppy0.score(xp_emb, xs_emb, xo_emb, facts=facts, entity_embeddings=entity_embeddings.weight)
        # (scores0_sp, subs0_sp), (scores0_po, subs0_po) = res0

        # res1 = hoppy1.forward(xp_emb, xs_emb, xo_emb, facts=facts, entity_embeddings=entity_embeddings)
        inf1 = hoppy1.score(xp_emb, xs_emb, xo_emb, facts=facts, entity_embeddings=entity_embeddings.weight)
        # (scores1_sp, subs1_sp), (scores1_po, subs1_po) = res1

        # res2 = hoppy2.forward(xp_emb, xs_emb, xo_emb, facts=facts, entity_embeddings=entity_embeddings)
        inf2 = hoppy2.score(xp_emb, xs_emb, xo_emb, facts=facts, entity_embeddings=entity_embeddings.weight)
        # (scores2_sp, subs2_sp), (scores2_po, subs2_po) = res2

        # res3 = hoppy3.forward(xp_emb, xs_emb, xo_emb, facts=facts, entity_embeddings=entity_embeddings)
        inf3 = hoppy3.score(xp_emb, xs_emb, xo_emb, facts=facts, entity_embeddings=entity_embeddings.weight)
        # (scores3_sp, subs3_sp), (scores3_po, subs3_po) = res3

        # res4 = hoppy4.forward(xp_emb, xs_emb, xo_emb, facts=facts, entity_embeddings=entity_embeddings)
        inf4 = hoppy4.score(xp_emb, xs_emb, xo_emb, facts=facts, entity_embeddings=entity_embeddings.weight)
        # (scores4_sp, subs4_sp), (scores4_po, subs4_po) = res4

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

        print(inf3_np)

        print(entity_embeddings.weight[entity_to_index['c'], 0].item())
        print(entity_embeddings.weight[entity_to_index['e'], 0].item())
        print(entity_embeddings.weight[entity_to_index['g'], 0].item())
        print(entity_embeddings.weight[entity_to_index['i'], 0].item())


if __name__ == '__main__':
    pytest.main([__file__])
    # test_clutrr_v2()
