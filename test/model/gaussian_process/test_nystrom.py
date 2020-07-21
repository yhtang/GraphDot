#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest.mock
import pytest
import numpy as np
from graphdot.model.gaussian_process import LowRankApproximateGPR

np.random.seed(0)


@pytest.mark.parametrize('X', [
    np.arange(25),
    np.linspace(0, 1, 25),
    np.linspace(-1, 1, 25),
])
@pytest.mark.parametrize('y', [
    np.zeros(25),
    -np.ones(25),
    np.ones(25),
    np.arange(25),
    np.sin(np.linspace(0, 1, 25) * 2 * np.pi),
    np.linspace(-1, 1, 25),
] + [
    np.random.rand(25) for _ in range(5)
] + [
    np.random.randn(25) for _ in range(5)
])
def test_nystrom_fit_self_consistency(X, y):

    class Kernel:
        def __call__(self, X, Y=None, s=0.01):
            return np.exp(
                -np.subtract.outer(X, Y if Y is not None else X)**2 / s**2
            )

        def diag(self, X):
            return np.ones_like(X)

    kernel = Kernel()
    core_idx = np.random.choice(len(X), 5, False)
    C = X[core_idx]
    c = y[core_idx]
    gpr = LowRankApproximateGPR(kernel=kernel, core=C, alpha=1e-7)
    with pytest.raises(RuntimeError):
        gpr.predict(X)
    gpr.fit(X, y)
    z = gpr.predict(C)
    assert(z == pytest.approx(c, 1e-3, 1e-3))
    z, std = gpr.predict(C, return_std=True)
    assert(z == pytest.approx(c, 1e-3, 1e-3))
    assert(std == pytest.approx(np.zeros_like(c), 1e-3, 1e-3))
    z, cov = gpr.predict(C, return_cov=True)
    assert(z == pytest.approx(c, 1e-3, 1e-3))
    assert(cov == pytest.approx(np.zeros((len(C), len(C))), 1e-3, 1e-3))


def test_nystrom_large_dataset():

    class Kernel:
        def __call__(self, X, Y=None, s=1.0):
            return np.exp(
                -np.subtract.outer(X, Y if Y is not None else X)**2 / s**2
            )

        def diag(self, X):
            return np.ones_like(X)

    kernel = Kernel()
    X = np.linspace(0, 1, 10000)
    Z = np.linspace(0, 1, 9999)
    y = np.sin(X * np.pi)
    z = np.sin(Z * np.pi)
    C = np.linspace(0, 1, 5)
    gpr = LowRankApproximateGPR(kernel=kernel, core=C, alpha=1e-7)
    gpr.fit(X, y)
    z_pred = gpr.predict(Z)
    assert(z_pred == pytest.approx(z, 1e-3, 1e-3))


# def test_gpr_predict_periodic():
#     '''test with a function with exactly two periods, and see if the GPR
#     can use information across the periods to fill in the missing points.'''

#     class Kernel:
#         def __call__(self, X, Y=None):
#             d = np.subtract.outer(X, Y if Y is not None else X)
#             return np.exp(-2 * np.sin(np.pi / 0.5 * d)**2)

#         def diag(self, X):
#             return np.ones_like(X)

#     kernel = Kernel()
#     X = np.linspace(0, 1, 16, endpoint=False)
#     y = np.sin(X * 4 * np.pi)
#     mask = np.array([1, 0, 1, 0, 1, 0, 1, 0,
#                      0, 1, 0, 1, 0, 1, 0, 1], dtype=np.bool_)

#     gpr = LowRankApproximateGPR(kernel=kernel, alpha=1e-10)
#     gpr.fit(X[mask], y[mask])
#     z = gpr.predict(X[~mask])
#     assert(z == pytest.approx(y[~mask], 1e-6))
