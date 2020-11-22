#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from scipy.stats import norm
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
        # print('x0', x0)
        # print('self.s', self.s)
        # print('self.n', self.n)
        return x0.state + self.s * np.random.randn(self.n)


# function to be learned
def f(x):
    # return np.sin(x) + 2e-4 * x**3 - 2.0 * np.exp(-x**2)
    return 2 * x


gpr = GaussianProcessRegressor(kernel=Kernel(0.5))
rewriter = RandomJitter(0.333, 9)
x = np.linspace(0, 3, 7)
y = f(x)
gpr.fit(x, y)
mcts = LikelihoodDrivenTreeSearch(rewriter, gpr, alpha=1e-10)
tree = mcts.search(seed=0.5, target=2.0)
flattree = tree.flat.to_pandas()
flattree['likelihood'] = norm.pdf(2.0, flattree.self_mean, flattree.self_std)
print(flattree.sort_values(['likelihood']))
