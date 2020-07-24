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


def test_nystrom_log_marginal_likelihood():

    class Kernel:
        def __init__(self, L):
            self.L = L

        def __call__(self, X, Y=None, eval_gradient=False):
            L = self.L
            d = np.subtract.outer(X, Y if Y is not None else X)
            f = np.exp(-0.5 * d**2 / L**2)
            if eval_gradient is False:
                return f
            else:
                j = np.exp(-0.5 * d**2 / L**2) * d**2 * L**-3
                return f, np.stack((j, ), axis=2)

        def diag(self, X):
            return np.ones_like(X)

        @property
        def theta(self):
            return np.log([self.L])

        @theta.setter
        def theta(self, t):
            self.L = np.exp(t[0])

        def clone_with_theta(self, theta):
            k = Kernel(1.0)
            k.theta = theta
            return k

    X = np.linspace(-1, 1, 24, endpoint=False)
    C = np.linspace(-1, 1, 6, endpoint=False)
    y = np.sin(X * np.pi)
    eps = 1e-3
    # for L in np.logspace(-1, 0.5, 10):
    for L in [1.0]:
        kernel = Kernel(L)
        gpr = LowRankApproximateGPR(kernel=kernel, core=C, alpha=1e-10)
        _, dL = gpr.log_marginal_likelihood(X=X, y=y, eval_gradient=True)
        theta0 = np.copy(kernel.theta)

        L_pos = gpr.log_marginal_likelihood(
            theta=theta0 + eps, X=X, y=y, eval_gradient=False
        ).item()
        L_neg = gpr.log_marginal_likelihood(
            theta=theta0 - eps, X=X, y=y, eval_gradient=False
        ).item()

        print('L_pos, L_neg', L_pos, L_neg)

        dL_diff = (L_pos - L_neg) / (2 * eps)
        assert(dL.item() == pytest.approx(dL_diff, 1e-3, 1e-3))


@pytest.mark.parametrize('cstride', [2])
def test_nystrom_predict_loocv(cstride):

    class Kernel:
        def __call__(self, X, Y=None):
            return np.exp(-np.subtract.outer(X, Y if Y is not None else X)**2)

        def diag(self, X):
            return np.ones_like(X)

    kernel = Kernel()
    X = np.linspace(-1, 1, 6)
    C = X[::cstride]
    gpr = LowRankApproximateGPR(kernel=kernel, core=C, alpha=1e-7)
    y = np.cos(X * np.pi)
    y_loocv, std_loocv = gpr.predict_loocv(X, y, return_std=True)
    assert(y_loocv == pytest.approx(gpr.predict_loocv(X, y, return_std=False)))
    for i, _ in enumerate(X):
        Xi = np.delete(X, i)
        yi = np.delete(y, i)
        gpr_loocv = LowRankApproximateGPR(kernel=kernel, core=C, alpha=1e-7)
        gpr_loocv.fit(Xi, yi)
        y_loocv_i, std_loocv_i = gpr_loocv.predict(X[[i]], return_std=True)
        assert(y_loocv_i.item() == pytest.approx(y_loocv[i], abs=1e-5))
        assert(std_loocv_i.item() == pytest.approx(std_loocv[i], abs=1e-5))


@pytest.mark.parametrize('normalize_y', [True, False])
def test_nystrom_fit_loocv_no_opt(normalize_y):

    class Kernel:
        def __init__(self, L):
            self.L = L

        def __call__(self, X, Y=None):
            L = self.L
            d = np.subtract.outer(X, Y if Y is not None else X)
            return np.exp(-0.5 * d**2 / L**2)

        def diag(self, X):
            return np.ones_like(X)

    X = np.linspace(-1, 1, 10, endpoint=False)
    C = np.linspace(-1, 1, 5, endpoint=False)
    y = np.sin(X * np.pi)
    kernel = Kernel(0.3)
    gpr = LowRankApproximateGPR(
        kernel=kernel,
        core=C,
        alpha=1e-7,
        normalize_y=normalize_y,
        optimizer=False,
    )
    _, m1, s1 = gpr.fit_loocv(X, y, return_mean=True, return_std=True)
    m2, s2 = gpr.predict_loocv(X, y, return_std=True)
    assert(m1 == pytest.approx(m2))
    assert(s1 == pytest.approx(s2))

