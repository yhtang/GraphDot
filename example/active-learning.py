#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from graphdot.model.gaussian_process import GaussianProcessRegressor
from graphdot.model.active_learning import (
    # DeterminantMaximizer,
    VarianceMinimizer,
    HierarchicalDrafter
)


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
# drafter = HierarchicalDrafter(DeterminantMaximizer(kernel))
drafter = HierarchicalDrafter(
    VarianceMinimizer(kernel),
    k=4,
    a=3
)
trials = [drafter(X, n) for _ in range(5)]
for i, trial in enumerate(trials):
    print(f'trial {i} det {np.prod(np.linalg.slogdet(kernel(X[trial])))}')
active_set = sorted(
    trials,
    key=lambda I: np.prod(np.linalg.slogdet(kernel(X[I])))
)[-1]
random_set = np.random.choice(len(X), n, False)


def test(chosen, label, color):
    gpr = GaussianProcessRegressor(kernel, normalize_y=True)
    gpr.fit(X[chosen], y[chosen])
    plt.scatter(X[chosen], y[chosen], label=label, color=color)
    plt.plot(grid, gpr.predict(grid), label=label, color=color)
    print(f"RMSE of '{label}' model:", np.std(gpr.predict(X) - y))
    print(f"{label} det: {np.prod(np.linalg.slogdet(kernel(X[chosen])))}")


plt.figure()
grid = np.linspace(X.min(), X.max(), 500)
plt.plot(grid, f(grid), lw=2, ls='dashed', color='k', alpha=0.5,
         label='ground_truth')
test(active_set, 'active', (0.0, 0.2, 0.7))
test(random_set, 'random', (0.9, 0.4, 0.0))
plt.legend()
plt.show()
