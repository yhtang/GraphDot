#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from graphdot.model.gaussian_process import GaussianProcessRegressor
from graphdot.model.active_learning import StochasticVolumeMaximizer


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


# function to be learned
def f(x):
    return np.sin(x) + 2e-4 * x**3 - 2.0 * np.exp(-x**2)


np.random.seed(0)
kernel = Kernel(s=2.0)
X = np.sort(np.random.randn(300) * 10)
y = f(X)
n = 40  # training set budget
vm = StochasticVolumeMaximizer(kernel)
active_set = vm(X, n)
random_set = np.random.choice(len(X), n, False)


def test(chosen, label, color):
    gpr = GaussianProcessRegressor(kernel, normalize_y=True)
    gpr.fit(X[chosen], y[chosen])
    plt.scatter(X[chosen], y[chosen], label=label, color=color)
    plt.plot(grid, gpr.predict(grid), label=label, color=color)
    print(f"RMSE of '{label}' model:", np.std(gpr.predict(X) - y))


plt.figure()
grid = np.linspace(X.min(), X.max(), 500)
plt.plot(grid, f(grid), lw=0.5, color='k', label='ground_truth')
test(active_set, 'active', (0.0, 0.2, 0.7))
test(random_set, 'random', (0.9, 0.4, 0.0))
plt.legend()
plt.show()
