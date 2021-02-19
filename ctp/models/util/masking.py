# -*- coding: utf-8 -*-

import torch
from torch import Tensor


def generate_kb_mask(indices: Tensor, batch_size: int, kb_size: int) -> Tensor:
    assert indices.shape[0] == batch_size
    mask = torch.ones([batch_size, kb_size], dtype=torch.float)
    mask[torch.arange(batch_size), indices] = 0
    return mask
