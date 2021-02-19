# -*- coding: utf-8 -*-

import numpy as np

import torch
from torch import nn, optim

from ctp.kernels import GaussianKernel
from ctp.models import NeuralKB
from ctp.regularizers import N3

from ctp.models.reasoning import SimpleHoppy
from ctp.reformulators import AttentiveReformulator

from ctp.util import make_batches

import pytest


@pytest.mark.light
def test_learning_v1():
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

    entity_embeddings = nn.Embedding(nb_entities, embedding_size * 2, sparse=True)
    predicate_embeddings = nn.Embedding(nb_predicates, embedding_size * 2, sparse=True)

    fact_rel = torch.from_numpy(np.array([predicate_to_index[p] for (_, p, _) in triples]))
    fact_arg1 = torch.from_numpy(np.array([entity_to_index[s] for (s, _, _) in triples]))
    fact_arg2 = torch.from_numpy(np.array([entity_to_index[o] for (_, _, o) in triples]))
    facts = [fact_rel, fact_arg1, fact_arg2]

    for st in ['min', 'concat']:
        model = NeuralKB(entity_embeddings=entity_embeddings, predicate_embeddings=predicate_embeddings,
                         kernel=kernel, facts=facts, scoring_type=st)

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

                        inf = model.score(xp_emb, xs_emb, xo_emb)

                        inf_np = inf.cpu().numpy()

                        if (s, p, o) in triples:
                            assert inf_np[0] > 0.95
                        else:
                            assert inf_np[0] < 0.01


@pytest.mark.light
def test_learning_v2():
    embedding_size = 100

    torch.manual_seed(0)

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

    entity_embeddings = nn.Embedding(nb_entities, embedding_size * 2, sparse=True)
    predicate_embeddings = nn.Embedding(nb_predicates, embedding_size * 2, sparse=True)

    fact_rel = torch.from_numpy(np.array([predicate_to_index[p] for (_, p, _) in triples]))
    fact_arg1 = torch.from_numpy(np.array([entity_to_index[s] for (s, _, _) in triples]))
    fact_arg2 = torch.from_numpy(np.array([entity_to_index[o] for (_, _, o) in triples]))
    facts = [fact_rel, fact_arg1, fact_arg2]

    model = NeuralKB(entity_embeddings=entity_embeddings, predicate_embeddings=predicate_embeddings,
                     kernel=kernel, facts=facts)

    reformulator = AttentiveReformulator(2, predicate_embeddings)
    hoppy = SimpleHoppy(model, entity_embeddings, hops=reformulator)

    for s, p, o in hops:
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

            inf = hoppy.score(xp_emb, xs_emb, xo_emb)

            inf_np = inf.cpu().numpy()
            assert inf_np < 0.5


@pytest.mark.light
def test_learning_v3():
    embedding_size = 10
    batch_size = 16

    triples, hops = [], []

    for i in range(16):
        triples += [(f'a{i}', 'p', f'b{i}'), (f'b{i}', 'q', f'c{i}')]
        hops += [(f'a{i}', 'r', f'c{i}')]

    entity_lst = sorted({e for (e, _, _) in triples + hops} | {e for (e, _, e) in triples + hops})
    predicate_lst = sorted({p for (_, p, _) in triples + hops})

    nb_entities, nb_predicates = len(entity_lst), len(predicate_lst)

    entity_to_index = {e: i for i, e in enumerate(entity_lst)}
    predicate_to_index = {p: i for i, p in enumerate(predicate_lst)}

    torch.manual_seed(0)

    kernel = GaussianKernel()

    entity_embeddings = nn.Embedding(nb_entities, embedding_size * 2, sparse=True)
    predicate_embeddings = nn.Embedding(nb_predicates, embedding_size * 2, sparse=True)

    fact_rel = torch.from_numpy(np.array([predicate_to_index[p] for (_, p, _) in triples]))
    fact_arg1 = torch.from_numpy(np.array([entity_to_index[s] for (s, _, _) in triples]))
    fact_arg2 = torch.from_numpy(np.array([entity_to_index[o] for (_, _, o) in triples]))
    facts = [fact_rel, fact_arg1, fact_arg2]

    model = NeuralKB(entity_embeddings=entity_embeddings, predicate_embeddings=predicate_embeddings,
                     kernel=kernel, facts=facts)

    reformulator = AttentiveReformulator(2, predicate_embeddings)
    hoppy = SimpleHoppy(model, entity_embeddings, hops=reformulator)

    N3_reg = N3()

    params = [p for p in hoppy.parameters()
              if not torch.equal(p, entity_embeddings.weight) and not torch.equal(p, predicate_embeddings.weight)]

    loss_function = nn.CrossEntropyLoss(reduction='mean')

    p_emb = predicate_embeddings(torch.from_numpy(np.array([predicate_to_index['p']])))
    q_emb = predicate_embeddings(torch.from_numpy(np.array([predicate_to_index['q']])))
    # r_emb = predicate_embeddings(torch.from_numpy(np.array([predicate_to_index['r']])))

    optimizer = optim.Adagrad(params, lr=0.1)

    hops_data = []
    for i in range(128):
        hops_data += hops

    batches = make_batches(len(hops_data), batch_size)

    c, d = 0.0, 0.0

    for batch_start, batch_end in batches:
        hops_batch = hops_data[batch_start:batch_end]

        s_lst = [s for (s, _, _) in hops_batch]
        p_lst = [p for (_, p, _) in hops_batch]
        o_lst = [o for (_, _, o) in hops_batch]

        xs_np = np.array([entity_to_index[s] for s in s_lst])
        xp_np = np.array([predicate_to_index[p] for p in p_lst])
        xo_np = np.array([entity_to_index[o] for o in o_lst])

        xs = torch.from_numpy(xs_np)
        xp = torch.from_numpy(xp_np)
        xo = torch.from_numpy(xo_np)

        xs_emb = entity_embeddings(xs)
        xp_emb = predicate_embeddings(xp)
        xo_emb = entity_embeddings(xo)

        sp_scores, po_scores = hoppy.forward(xp_emb, xs_emb, xo_emb)

        loss = loss_function(sp_scores, xo) + loss_function(po_scores, xs)

        factors = [hoppy.factor(e) for e in [xp_emb, xs_emb, xo_emb]]
        loss += 0.1 * N3_reg(factors)

        tmp = hoppy.hops(xp_emb)
        hop_1_emb = tmp[0]
        hop_2_emb = tmp[1]

        c = kernel.pairwise(p_emb, hop_1_emb).mean().cpu().detach().numpy()
        d = kernel.pairwise(q_emb, hop_2_emb).mean().cpu().detach().numpy()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    assert c > 0.95
    assert d > 0.95


if __name__ == '__main__':
    pytest.main([__file__])
    # test_learning_v3()
