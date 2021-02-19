# -*- coding: utf-8 -*-

from torch.optim.optimizer import Optimizer


class Optimizers:
    def __init__(self, *op: Optimizer):
        self.optimizers = op

    def zero_grad(self):
        for op in self.optimizers:
            op.zero_grad()

    def step(self):
        for op in self.optimizers:
            op.step()
