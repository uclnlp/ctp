# -*- coding: utf-8 -*-

import multiprocessing

import numpy as np
import nmslib

from kbcr.indexing.base import Index

from typing import Optional


class NMSSearchIndex(Index):
    def __init__(self,
                 method: str = 'hnsw',
                 space: str = 'l2',
                 num_threads: Optional[int] = None,
                 m: int = 15,
                 efc: int = 100,
                 efs: int = 100):
        super().__init__()
        self.method = method
        self.space = space
        self.num_threads = num_threads if num_threads is not None else multiprocessing.cpu_count()
        self.m = m
        self.efc = efc
        self.efs = efs

        self.index = None

    def build(self,
              data: np.ndarray):
        self.index = nmslib.init(method=self.method, space=self.space,
                                 data_type=nmslib.DataType.DENSE_VECTOR)
        self.index.addDataPointBatch(data)

        index_params = {
            'M': self.m,
            'indexThreadQty': self.num_threads,
            'efConstruction': self.efc,
            'post': 0
        }

        query_params = {
            'efSearch': self.efs
        }

        self.index.createIndex(index_params, print_progress=False)
        self.index.setQueryTimeParams(query_params)

    def query(self,
              data: np.ndarray,
              k: int = 5) -> np.ndarray:
        neighbours = self.index.knnQueryBatch(data, k=k, num_threads=self.num_threads)
        res = np.array([index for index, _ in neighbours])
        return res

