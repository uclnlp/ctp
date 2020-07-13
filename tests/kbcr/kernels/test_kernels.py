# -*- coding: utf-8 -*-

import numpy as np

import torch
from torch import nn

from kbcr.kernels import GaussianKernel

import pytest


@pytest.mark.light
def test_gaussian_v1():
    nb_entities = 10
    embedding_size = 20
    slope = 1.0

    seed = 0

    np.random.seed(seed)
    torch.manual_seed(seed)

    with torch.no_grad():
        x_emb = nn.Embedding(nb_entities, embedding_size, sparse=True)
        y_emb = nn.Embedding(nb_entities, embedding_size, sparse=True)

        x_emb.weight.data *= 1e-3
        y_emb.weight.data *= 1e-3

        kernel = GaussianKernel(slope=slope)

        a = kernel(x_emb.weight, y_emb.weight)
        b = kernel(x_emb.weight, x_emb.weight)

        c = kernel.pairwise(x_emb.weight, y_emb.weight)
        d = kernel.pairwise(x_emb.weight, x_emb.weight)

        a_np = a.numpy()
        b_np = b.numpy()
        c_np = c.numpy()
        d_np = d.numpy()

        np.testing.assert_allclose(a_np, np.diag(c_np), rtol=1e-7, atol=1e-7)
        np.testing.assert_allclose(b_np, np.diag(d_np), rtol=1e-7, atol=1e-7)


if __name__ == '__main__':
    pytest.main([__file__])
