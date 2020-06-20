#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest
import numpy as np
from graphdot.model.gaussian_process import GaussianProcessRegressor

np.random.seed(0)


@pytest.mark.parametrize('X', [
    np.arange(5),
    np.linspace(0, 1, 5),
    np.linspace(-1, 1, 5),
] + [
    np.random.rand(5) * 2.0 for _ in range(5)
])
@pytest.mark.parametrize('y', [
    np.zeros(5),
    -np.ones(5),
    np.ones(5),
    np.arange(5),
    np.sin(np.linspace(0, 1, 5) * 2 * np.pi),
    np.linspace(-1, 1, 5),
] + [
    np.random.rand(5) for _ in range(5)
] + [
    np.random.randn(5) for _ in range(5)
])
def test_gpr_fit_self_consistency(X, y):

    class Kernel:
        def __call__(self, X, Y=None):
            return np.exp(-np.subtract.outer(X, Y if Y is not None else X)**2)

        def diag(self, X):
            return np.ones_like(X)

    kernel = Kernel()
    gpr = GaussianProcessRegressor(kernel=kernel, alpha=1e-12)
    gpr.fit(X, y)
    assert(gpr.predict(X) == pytest.approx(y, 1e-3, 1e-3))


def test_gpr_periodic_regression():
    '''test with a function with exactly two periods, and see if the GPR
    can use information across the periods to fill in the missing points.'''

    class Kernel:
        def __call__(self, X, Y=None):
            d = np.subtract.outer(X, Y if Y is not None else X)
            return np.exp(-2 * np.sin(np.pi / 0.5 * d)**2)

        def diag(self, X):
            return np.ones_like(X)

    kernel = Kernel()
    X = np.linspace(0, 1, 16, endpoint=False)
    y = np.sin(X * 4 * np.pi)
    mask = np.array([1, 0, 1, 0, 1, 0, 1, 0,
                     0, 1, 0, 1, 0, 1, 0, 1], dtype=np.bool_)

    gpr = GaussianProcessRegressor(kernel=kernel, alpha=1e-10)
    gpr.fit(X[mask], y[mask])
    z = gpr.predict(X[~mask])
    assert(z == pytest.approx(y[~mask], 1e-6))


def test_gpr_hyperparameter_optimization():
    '''test with a function with exactly two periods, and see if the GPR
    can identify the frequency via hyperparameter optimization.'''

    class Kernel:
        def __init__(self, p, L):
            self.p = p
            self.L = L

        def __call__(self, X, Y=None, eval_gradient=False):
            d = np.subtract.outer(X, Y if Y is not None else X)
            f = np.exp(-2 * np.sin(np.pi / self.p * d)**2 / self.L**2)
            if eval_gradient is False:
                return f
            else:
                s = np.sin(d * np.pi / self.p)
                c = np.cos(d * np.pi / self.p)
                j1 = 2.0 * np.pi * d * 2 * s * c * f / self.p**2 / self.L**2
                j2 = 4.0 * s**2 * f / self.L**3
                return f, np.stack((j1, j2), axis=2)

        def diag(self, X):
            return np.ones_like(X)

        @property
        def theta(self):
            return np.log([self.p, self.L])

        @theta.setter
        def theta(self, t):
            self.p, self.L = np.exp(t)

        @property
        def bounds(self):
            return np.log([[1e-2, 10], [1e-2, 10]])

    X = np.linspace(0, 1, 16, endpoint=False)
    y = np.sin(X * 4 * np.pi)
    kernel = Kernel(0.49, 0.1)
    gpr = GaussianProcessRegressor(kernel=kernel, alpha=1e-10, optimizer=True)
    gpr.fit(X, y)
    assert(kernel.p == pytest.approx(0.5, 1e-2))
