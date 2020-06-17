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
    grid = np.linspace(0, 1, 16, endpoint=False)
    y = np.sin(grid * 4 * np.pi)

    mask = np.array([1, 0, 1, 0, 1, 0, 1, 0,
                     0, 1, 0, 1, 0, 1, 0, 1], dtype=np.bool_)

    gpr = GaussianProcessRegressor(kernel=kernel, alpha=1e-10)

    gpr.fit(grid[mask], y[mask])
    z = gpr.predict(grid[~mask])
    assert(z == pytest.approx(y[~mask], 1e-6))
