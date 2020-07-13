# -*- coding: utf-8 -*-

import numpy as np

from kbcr.indexing.base import Index


class NPSearchIndex(Index):
    def __init__(self):
        super().__init__()
        self.data = None

    def build(self,
              data: np.ndarray):
        self.data = data

    def query(self,
              data: np.ndarray,
              k: int = 5) -> np.ndarray:
        nb_instances = data.shape[0]
        res = []
        for i in range(nb_instances):
            sqd = np.sqrt(((self.data - data[i, :]) ** 2).sum(axis=1))
            indices = np.argsort(sqd)
            top_k_indices = indices[:k].tolist()
            res += [top_k_indices]
        res = np.array(res)
        return res

