#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from graphdot.model.mcts.ldgr import LikelihoodDrivenTreeSearch
from graphdot.model.gaussian_process import GaussianProcessRegressor


# Good old Gaussian/RBF kernel
class Kernel:
    def __init__(self, s):
        self.s = s

    def __call__(self, X, Y=None):
        return np.exp(
            -np.subtract.outer(X, Y if Y is not None else X)**2 / self.s**2
        )

    def diag(self, X):
        return np.ones_like(X)


class RandomJitter:
    def __init__(self, s, n):
        self.s = s
        self.n = n

    def __call__(self, x0):
        print('x0', x0)
        print('self.s', self.s)
        print('self.n', self.n)
        return x0.label + self.s * np.random.randn(self.n)


gpr = GaussianProcessRegressor(kernel=Kernel(1.0))
rewriter = RandomJitter(0.1, 4)

# function to be learned
def f(x):
    return np.sin(x) + 2e-4 * x**3 - 2.0 * np.exp(-x**2)


x = np.linspace(0, 3, 7)
y = f(x)
gpr.fit(x, y)


# rewriter = SMILESRewriter(pool='default')
mcts = LikelihoodDrivenTreeSearch(rewriter, gpr, alpha=1e-10)

# result = mcts('C', branching_factor=16, target=-824.9)

mcts(1.2, 2.0)
