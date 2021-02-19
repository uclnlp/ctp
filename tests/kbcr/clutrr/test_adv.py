# -*- coding: utf-8 -*-

import numpy as np

import multiprocessing

import torch
from torch import nn, Tensor

from ctp.kernels import GaussianKernel
from ctp.clutrr.models import BatchNeuralKB, BatchHoppy, BatchUnary, BatchMulti
from ctp.reformulators import SymbolicReformulator

from typing import List, Dict, Tuple, Optional

import pytest

torch.set_num_threads(multiprocessing.cpu_count())


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
def test_adv_v1():
    embedding_size = 20

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

        batch_size = 12
        fact_size = rel_emb.shape[0]
        entity_size = entity_embeddings.weight.shape[0]

        rel_emb = rel_emb.view(1, fact_size, -1).repeat(batch_size, 1, 1)
        arg1_emb = arg1_emb.view(1, fact_size, -1).repeat(batch_size, 1, 1)
        arg2_emb = arg2_emb.view(1, fact_size, -1).repeat(batch_size, 1, 1)
        nb_facts = torch.tensor([fact_size for _ in range(batch_size)], dtype=torch.long)

        emb = entity_embeddings.weight.view(1, entity_size, -1).repeat(batch_size, 1, 1)
        _nb_entities = torch.tensor([entity_size for _ in range(batch_size)], dtype=torch.long)

        facts = [rel_emb, arg1_emb, arg2_emb]

        model = BatchNeuralKB(kernel=kernel)

        indices = torch.from_numpy(np.array([predicate_to_index['p'], predicate_to_index['q']]))
        reformulator = SymbolicReformulator(predicate_embeddings, indices)

        hoppy0 = BatchHoppy(model, hops_lst=[(reformulator, False)], depth=0)
        hoppy1 = BatchHoppy(model, hops_lst=[(reformulator, False)], depth=1)
        hoppy2 = BatchHoppy(model, hops_lst=[(reformulator, False)], depth=2)
        hoppy3 = BatchHoppy(model, hops_lst=[(reformulator, False)], depth=3)

        xs_np = rs.randint(nb_entities, size=batch_size)
        xp_np = rs.randint(nb_predicates, size=batch_size)
        xo_np = rs.randint(nb_entities, size=batch_size)

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

        inf0 = hoppy0.score(xp_emb, xs_emb, xo_emb, facts=facts, nb_facts=nb_facts,
                            entity_embeddings=emb, nb_entities=_nb_entities)

        inf1 = hoppy1.score(xp_emb, xs_emb, xo_emb, facts=facts, nb_facts=nb_facts,
                            entity_embeddings=emb, nb_entities=_nb_entities)

        inf2 = hoppy2.score(xp_emb, xs_emb, xo_emb, facts=facts, nb_facts=nb_facts,
                            entity_embeddings=emb, nb_entities=_nb_entities)

        inf3 = hoppy3.score(xp_emb, xs_emb, xo_emb, facts=facts, nb_facts=nb_facts,
                            entity_embeddings=emb, nb_entities=_nb_entities)

        inf0_np = inf0.cpu().numpy()
        inf1_np = inf1.cpu().numpy()
        inf2_np = inf2.cpu().numpy()
        inf3_np = inf3.cpu().numpy()

        np.testing.assert_allclose(inf0_np, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], rtol=1e-1, atol=1e-1)
        np.testing.assert_allclose(inf1_np, [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], rtol=1e-1, atol=1e-1)
        np.testing.assert_allclose(inf2_np, [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], rtol=1e-1, atol=1e-1)
        np.testing.assert_allclose(inf3_np, [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0], rtol=1e-1, atol=1e-1)

        print(inf3_np)


@pytest.mark.light
def test_adv_v2():
    embedding_size = 20

    torch.manual_seed(0)
    rs = np.random.RandomState(0)

    triples = [
        ('a', 'p', 'b'),
        ('a', 'p', 'd'),
        ('c', 'p', 'd'),
        ('e', 'q', 'f'),
        ('f', 'p', 'c'),
        ('x', 'r', 'y')
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

        batch_size = 6
        fact_size = rel_emb.shape[0]
        entity_size = entity_embeddings.weight.shape[0]

        rel_emb = rel_emb.view(1, fact_size, -1).repeat(batch_size, 1, 1)
        arg1_emb = arg1_emb.view(1, fact_size, -1).repeat(batch_size, 1, 1)
        arg2_emb = arg2_emb.view(1, fact_size, -1).repeat(batch_size, 1, 1)
        nb_facts = torch.tensor([fact_size for _ in range(batch_size)], dtype=torch.long)

        emb = entity_embeddings.weight.view(1, entity_size, -1).repeat(batch_size, 1, 1)
        _nb_entities = torch.tensor([entity_size for _ in range(batch_size)], dtype=torch.long)

        facts = [rel_emb, arg1_emb, arg2_emb]

        model = BatchNeuralKB(kernel=kernel)

        indices = torch.from_numpy(np.array([predicate_to_index['p']]))
        reformulator = SymbolicReformulator(predicate_embeddings, indices)

        unary = BatchUnary(model, hops_lst=[(reformulator, False)])

        xs_np = rs.randint(nb_entities, size=batch_size)
        xp_np = rs.randint(nb_predicates, size=batch_size)
        xo_np = rs.randint(nb_entities, size=batch_size)

        xs_np[0] = entity_to_index['a']
        xp_np[0] = predicate_to_index['r']
        xo_np[0] = entity_to_index['a']

        xs_np[1] = entity_to_index['a']
        xp_np[1] = predicate_to_index['r']
        xo_np[1] = entity_to_index['b']

        xs_np[2] = entity_to_index['a']
        xp_np[2] = predicate_to_index['r']
        xo_np[2] = entity_to_index['c']

        xs_np[3] = entity_to_index['a']
        xp_np[3] = predicate_to_index['r']
        xo_np[3] = entity_to_index['d']

        xs_np[4] = entity_to_index['a']
        xp_np[4] = predicate_to_index['r']
        xo_np[4] = entity_to_index['e']

        xs_np[5] = entity_to_index['a']
        xp_np[5] = predicate_to_index['r']
        xo_np[5] = entity_to_index['f']

        xs = torch.from_numpy(xs_np)
        xp = torch.from_numpy(xp_np)
        xo = torch.from_numpy(xo_np)

        xs_emb = entity_embeddings(xs)
        xp_emb = predicate_embeddings(xp)
        xo_emb = entity_embeddings(xo)

        inf = unary.score(xp_emb, xs_emb, xo_emb, facts=facts, nb_facts=nb_facts,
                          entity_embeddings=emb, nb_entities=_nb_entities)

        inf_np = inf.cpu().numpy()

        print(inf_np)

        np.testing.assert_allclose(inf_np, [1] * batch_size, rtol=1e-2, atol=1e-2)


@pytest.mark.light
def test_adv_v3():
    embedding_size = 20

    torch.manual_seed(0)
    rs = np.random.RandomState(0)

    triples = [
        ('a', 'p', 'b'),
        ('a', 'p', 'd'),
        ('c', 'p', 'd'),
        ('e', 'q', 'f'),
        ('f', 'p', 'c'),
        ('x', 'r', 'y')
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

        batch_size = 6
        fact_size = rel_emb.shape[0]
        entity_size = entity_embeddings.weight.shape[0]

        rel_emb = rel_emb.view(1, fact_size, -1).repeat(batch_size, 1, 1)
        arg1_emb = arg1_emb.view(1, fact_size, -1).repeat(batch_size, 1, 1)
        arg2_emb = arg2_emb.view(1, fact_size, -1).repeat(batch_size, 1, 1)
        nb_facts = torch.tensor([fact_size for _ in range(batch_size)], dtype=torch.long)

        emb = entity_embeddings.weight.view(1, entity_size, -1).repeat(batch_size, 1, 1)
        _nb_entities = torch.tensor([entity_size for _ in range(batch_size)], dtype=torch.long)

        facts = [rel_emb, arg1_emb, arg2_emb]

        model = BatchNeuralKB(kernel=kernel)

        indices = torch.from_numpy(np.array([predicate_to_index['p']]))
        reformulator = SymbolicReformulator(predicate_embeddings, indices)

        unary = BatchUnary(model, hops_lst=[(reformulator, True)])

        xs_np = rs.randint(nb_entities, size=batch_size)
        xp_np = rs.randint(nb_predicates, size=batch_size)
        xo_np = rs.randint(nb_entities, size=batch_size)

        xs_np[0] = entity_to_index['a']
        xp_np[0] = predicate_to_index['r']
        xo_np[0] = entity_to_index['a']

        xs_np[1] = entity_to_index['a']
        xp_np[1] = predicate_to_index['r']
        xo_np[1] = entity_to_index['b']

        xs_np[2] = entity_to_index['a']
        xp_np[2] = predicate_to_index['r']
        xo_np[2] = entity_to_index['c']

        xs_np[3] = entity_to_index['a']
        xp_np[3] = predicate_to_index['r']
        xo_np[3] = entity_to_index['d']

        xs_np[4] = entity_to_index['a']
        xp_np[4] = predicate_to_index['r']
        xo_np[4] = entity_to_index['e']

        xs_np[5] = entity_to_index['a']
        xp_np[5] = predicate_to_index['r']
        xo_np[5] = entity_to_index['f']

        xs = torch.from_numpy(xs_np)
        xp = torch.from_numpy(xp_np)
        xo = torch.from_numpy(xo_np)

        xs_emb = entity_embeddings(xs)
        xp_emb = predicate_embeddings(xp)
        xo_emb = entity_embeddings(xo)

        inf = unary.score(xp_emb, xs_emb, xo_emb, facts=facts, nb_facts=nb_facts,
                          entity_embeddings=emb, nb_entities=_nb_entities)

        inf_np = inf.cpu().numpy()

        print(inf_np)

        np.testing.assert_allclose(inf_np, [0] * batch_size, rtol=1e-2, atol=1e-2)


@pytest.mark.light
def test_adv_v4():
    embedding_size = 20

    torch.manual_seed(0)
    rs = np.random.RandomState(0)

    triples = [
        ('a', 'p', 'b'),
        ('a', 'p', 'd'),
        ('c', 'p', 'd'),
        ('e', 'q', 'f'),
        ('f', 'p', 'c'),
        ('x', 'r', 'y')
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

        batch_size = 6
        fact_size = rel_emb.shape[0]
        entity_size = entity_embeddings.weight.shape[0]

        rel_emb = rel_emb.view(1, fact_size, -1).repeat(batch_size, 1, 1)
        arg1_emb = arg1_emb.view(1, fact_size, -1).repeat(batch_size, 1, 1)
        arg2_emb = arg2_emb.view(1, fact_size, -1).repeat(batch_size, 1, 1)
        nb_facts = torch.tensor([fact_size for _ in range(batch_size)], dtype=torch.long)

        emb = entity_embeddings.weight.view(1, entity_size, -1).repeat(batch_size, 1, 1)
        _nb_entities = torch.tensor([entity_size for _ in range(batch_size)], dtype=torch.long)

        facts = [rel_emb, arg1_emb, arg2_emb]

        model = BatchNeuralKB(kernel=kernel)

        indices = torch.from_numpy(np.array([predicate_to_index['p']]))
        reformulator = SymbolicReformulator(predicate_embeddings, indices)

        unary = BatchUnary(model, hops_lst=[(reformulator, False)])

        xs_np = rs.randint(nb_entities, size=batch_size)
        xp_np = rs.randint(nb_predicates, size=batch_size)
        xo_np = rs.randint(nb_entities, size=batch_size)

        xs_np[0] = entity_to_index['b']
        xp_np[0] = predicate_to_index['r']
        xo_np[0] = entity_to_index['a']

        xs_np[1] = entity_to_index['b']
        xp_np[1] = predicate_to_index['r']
        xo_np[1] = entity_to_index['b']

        xs_np[2] = entity_to_index['b']
        xp_np[2] = predicate_to_index['r']
        xo_np[2] = entity_to_index['c']

        xs_np[3] = entity_to_index['b']
        xp_np[3] = predicate_to_index['r']
        xo_np[3] = entity_to_index['d']

        xs_np[4] = entity_to_index['b']
        xp_np[4] = predicate_to_index['r']
        xo_np[4] = entity_to_index['e']

        xs_np[5] = entity_to_index['b']
        xp_np[5] = predicate_to_index['r']
        xo_np[5] = entity_to_index['f']

        xs = torch.from_numpy(xs_np)
        xp = torch.from_numpy(xp_np)
        xo = torch.from_numpy(xo_np)

        xs_emb = entity_embeddings(xs)
        xp_emb = predicate_embeddings(xp)
        xo_emb = entity_embeddings(xo)

        inf = unary.score(xp_emb, xs_emb, xo_emb, facts=facts, nb_facts=nb_facts,
                          entity_embeddings=emb, nb_entities=_nb_entities)

        inf_np = inf.cpu().numpy()

        print(inf_np)

        np.testing.assert_allclose(inf_np, [0] * batch_size, rtol=1e-2, atol=1e-2)


@pytest.mark.light
def test_adv_v5():
    embedding_size = 20

    torch.manual_seed(0)
    rs = np.random.RandomState(0)

    triples = [
        ('a', 'p', 'b'),
        ('a', 'p', 'd'),
        ('c', 'p', 'd'),
        ('e', 'q', 'f'),
        ('f', 'p', 'c'),
        ('x', 'r', 'y')
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

        batch_size = 6
        fact_size = rel_emb.shape[0]
        entity_size = entity_embeddings.weight.shape[0]

        rel_emb = rel_emb.view(1, fact_size, -1).repeat(batch_size, 1, 1)
        arg1_emb = arg1_emb.view(1, fact_size, -1).repeat(batch_size, 1, 1)
        arg2_emb = arg2_emb.view(1, fact_size, -1).repeat(batch_size, 1, 1)
        nb_facts = torch.tensor([fact_size for _ in range(batch_size)], dtype=torch.long)

        emb = entity_embeddings.weight.view(1, entity_size, -1).repeat(batch_size, 1, 1)
        _nb_entities = torch.tensor([entity_size for _ in range(batch_size)], dtype=torch.long)

        facts = [rel_emb, arg1_emb, arg2_emb]

        model = BatchNeuralKB(kernel=kernel)

        indices = torch.from_numpy(np.array([predicate_to_index['p']]))
        reformulator = SymbolicReformulator(predicate_embeddings, indices)

        unary = BatchUnary(model, hops_lst=[(reformulator, True)])

        xs_np = rs.randint(nb_entities, size=batch_size)
        xp_np = rs.randint(nb_predicates, size=batch_size)
        xo_np = rs.randint(nb_entities, size=batch_size)

        xs_np[0] = entity_to_index['b']
        xp_np[0] = predicate_to_index['r']
        xo_np[0] = entity_to_index['a']

        xs_np[1] = entity_to_index['b']
        xp_np[1] = predicate_to_index['r']
        xo_np[1] = entity_to_index['b']

        xs_np[2] = entity_to_index['b']
        xp_np[2] = predicate_to_index['r']
        xo_np[2] = entity_to_index['c']

        xs_np[3] = entity_to_index['b']
        xp_np[3] = predicate_to_index['r']
        xo_np[3] = entity_to_index['d']

        xs_np[4] = entity_to_index['b']
        xp_np[4] = predicate_to_index['r']
        xo_np[4] = entity_to_index['e']

        xs_np[5] = entity_to_index['b']
        xp_np[5] = predicate_to_index['r']
        xo_np[5] = entity_to_index['f']

        xs = torch.from_numpy(xs_np)
        xp = torch.from_numpy(xp_np)
        xo = torch.from_numpy(xo_np)

        xs_emb = entity_embeddings(xs)
        xp_emb = predicate_embeddings(xp)
        xo_emb = entity_embeddings(xo)

        inf = unary.score(xp_emb, xs_emb, xo_emb, facts=facts, nb_facts=nb_facts,
                          entity_embeddings=emb, nb_entities=_nb_entities)

        inf_np = inf.cpu().numpy()

        print(inf_np)

        np.testing.assert_allclose(inf_np, [1] * batch_size, rtol=1e-2, atol=1e-2)


if __name__ == '__main__':
    pytest.main([__file__])
    # test_adv_v2()
    # test_adv_v3()
    # test_adv_v4()
    # test_adv_v5()
