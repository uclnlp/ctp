# -*- coding: utf-8 -*-

import torch
import numpy as np

from kbcr.indexing.base import Index

try:
    import faiss
except ImportError:
    from kbcr.indexing.nms import NMSSearchIndex

    class FAISSSearchIndex(NMSSearchIndex):
        pass
else:
    class FAISSSearchIndex(Index):
        def __init__(self,
                     use_gpu: bool = torch.cuda.is_available()):
            super().__init__()
            self.use_gpu = use_gpu

            self.index = self.res = None
            if self.use_gpu is True:
                self.res = faiss.StandardGpuResources()

        def build(self,
                  data: np.ndarray):
            k = data.shape[1]
            if self.use_gpu is True:
                self.index = faiss.GpuIndexFlatL2(self.res, k)
            else:
                self.index = faiss.IndexFlatL2(k)
            self.index.add(data)

        def query(self,
                  data: np.ndarray,
                  k: int = 5) -> np.ndarray:
            _, neighbour_indices = self.index.search(data, k)
            return neighbour_indices
