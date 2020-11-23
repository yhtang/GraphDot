#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from graphdot.model.tree_search import MCTSGraphTransformer
from graphdot.model.tree_search import AbstractRewriter
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


# Dummy rewriter of a real numbers
class RandomJitter(AbstractRewriter):
    def __init__(self, s, n):
        self.s = s
        self.n = n

    def __call__(self, x, rng):
        return np.minimum(3, np.maximum(0, rng.normal(x.g, self.s, self.n)))


# Function to be learned
def f(x):
    return np.sin(x) + 2e-4 * x**3 - 2.0 * np.exp(-x**2)
    # return 2 * x


# GPR surrogate model
x = np.linspace(0, 3, 13)
y = f(x)
gpr = GaussianProcessRegressor(kernel=Kernel(0.5))
gpr.fit(x, y)

# Run MCTS
mcts = MCTSGraphTransformer(
    rewriter=RandomJitter(0.333, 9),
    surrogate=gpr,
)
# print(mcts.seek(g0=0.5, target=2.0))
print(mcts.seek(g0=0.5, target=0.0, maxiter=20))
