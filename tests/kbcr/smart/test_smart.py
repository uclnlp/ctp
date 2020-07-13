# -*- coding: utf-8 -*-

import numpy as np

import torch
from torch import nn

from kbcr.kernels import GaussianKernel
from kbcr.smart import NeuralKB

import pytest


@pytest.mark.light
def test_smart_v1():
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

            fact_rel = torch.from_numpy(np.array([predicate_to_index[p] for (_, p, _) in triples]))
            fact_arg1 = torch.from_numpy(np.array([entity_to_index[s] for (s, _, _) in triples]))
            fact_arg2 = torch.from_numpy(np.array([entity_to_index[o] for (_, _, o) in triples]))
            facts = [fact_rel, fact_arg1, fact_arg2]

            model = NeuralKB(entity_embeddings=entity_embeddings, predicate_embeddings=predicate_embeddings,
                             kernel=kernel, facts=facts)

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

            res_sp, res_po = model.forward(xp_emb, xs_emb, xo_emb)
            inf = model.score(xp_emb, xs_emb, xo_emb)

            assert inf[0] > 0.9
            assert inf[1] > 0.9

            scores_sp, emb_sp = res_sp
            scores_po, emb_po = res_po

            print(scores_sp.shape, emb_sp.shape)
            print(scores_po.shape, emb_po.shape)

            inf = inf.cpu().numpy()
            scores_sp = scores_sp.cpu().numpy()
            scores_po = scores_po.cpu().numpy()

            print('AAA', inf)
            print('BBB', scores_sp)


if __name__ == '__main__':
    pytest.main([__file__])
    # test_smart_v1()
