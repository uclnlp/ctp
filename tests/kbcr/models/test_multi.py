# -*- coding: utf-8 -*-

import numpy as np

import torch
from torch import nn

from ctp.models import ComplEx, Multi
from ctp.models.reasoning import SimpleHoppy
from ctp.reformulators import LinearReformulator, AttentiveReformulator

import pytest


@pytest.mark.light
def test_multi():
    nb_entities = 10
    nb_predicates = 5
    embedding_size = 10

    init_size = 1.0

    rs = np.random.RandomState(0)

    for _ in range(8):
        for nb_hops in range(1, 6):
            for use_attention in [True, False]:
                for pt in {'max', 'min', 'sum', 'mixture'}:
                    with torch.no_grad():
                        entity_embeddings = nn.Embedding(nb_entities, embedding_size * 2, sparse=True)
                        predicate_embeddings = nn.Embedding(nb_predicates, embedding_size * 2, sparse=True)

                        entity_embeddings.weight.data *= init_size
                        predicate_embeddings.weight.data *= init_size

                        base = ComplEx(entity_embeddings)

                        models = []
                        for i in range(nb_hops):
                            if use_attention:
                                reformulator = AttentiveReformulator(i, predicate_embeddings)
                            else:
                                reformulator = LinearReformulator(i, embedding_size * 2)

                            h_model = SimpleHoppy(base, entity_embeddings, hops=reformulator)
                            models += [h_model]

                        model = Multi(models=models, pooling_type=pt, embedding_size=embedding_size * 2)

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
