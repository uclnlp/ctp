# -*- coding: utf-8 -*-

import numpy as np
import torch

from kbcr.models.util.masking import generate_kb_mask

import pytest


@pytest.mark.light
def test_kb_v1():
    indices = torch.arange(5)
    mask = generate_kb_mask(indices=indices + 1, batch_size=5, kb_size=8)

    np_mask = np.array([
        [1, 0, 1, 1, 1, 1, 1, 1],
        [1, 1, 0, 1, 1, 1, 1, 1],
        [1, 1, 1, 0, 1, 1, 1, 1],
        [1, 1, 1, 1, 0, 1, 1, 1],
        [1, 1, 1, 1, 1, 0, 1, 1]
    ])

    np.testing.assert_allclose(mask.numpy(), np_mask)


if __name__ == '__main__':
    pytest.main([__file__])
