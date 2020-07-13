# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
import numpy as np


class Index(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def build(self,
              data: np.ndarray):
        raise NotImplementedError

    @abstractmethod
    def query(self,
              data: np.ndarray,
              k: int = 5) -> np.ndarray:
        raise NotImplementedError
