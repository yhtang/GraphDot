#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest.mock
import pytest
import numpy as np
import os
import tempfile
from graphdot.model.gaussian_process import GPRNoiseDetector

np.random.seed(0)


# @pytest.mark.parametrize('X', [
#     np.arange(5),
#     np.linspace(0, 1, 5),
#     np.linspace(-1, 1, 5),
# ] + [
#     np.random.rand(5) * 2.0 for _ in range(5)
# ])
# @pytest.mark.parametrize('y', [
#     np.zeros(5),
#     -np.ones(5),
#     np.ones(5),
#     np.arange(5),
#     np.sin(np.linspace(0, 1, 5) * 2 * np.pi),
#     np.linspace(-1, 1, 5),
# ] + [
#     np.random.rand(5) for _ in range(5)
# ] + [
#     np.random.randn(5) for _ in range(5)
# ])
# def test_gpr_fit_self_consistency(X, y):

#     class Kernel:
#         def __call__(self, X, Y=None):
#             return np.exp(-np.subtract.outer(X, Y if Y is not None else X)**2)

#         def diag(self, X):
#             return np.ones_like(X)

#     kernel = Kernel()
#     gpr = GPRNoiseDetector(kernel=kernel, alpha=1e-12)
#     with pytest.raises(RuntimeError):
#         gpr.predict(X)
#     gpr.fit(X, y)
#     z = gpr.predict(X)
#     assert(z == pytest.approx(y, 1e-3, 1e-3))
#     z, std = gpr.predict(X, return_std=True)
#     assert(z == pytest.approx(y, 1e-3, 1e-3))
#     assert(std == pytest.approx(np.zeros_like(y), 1e-3, 1e-3))
#     z, cov = gpr.predict(X, return_cov=True)
#     assert(z == pytest.approx(y, 1e-3, 1e-3))
#     assert(cov == pytest.approx(np.zeros((len(X), len(X))), 1e-3, 1e-3))


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

#     gpr = GPRNoiseDetector(kernel=kernel, alpha=1e-10)
#     gpr.fit(X[mask], y[mask])
#     z = gpr.predict(X[~mask])
#     assert(z == pytest.approx(y[~mask], 1e-6))


# @pytest.mark.parametrize('repeat', [1, 3])
# @pytest.mark.parametrize('verbose', [True, False])
# def test_gpr_fit_mle(repeat, verbose):
#     '''test with a function with exactly two periods, and see if the GPR
#     can identify the frequency via hyperparameter optimization.'''

#     class Kernel:
#         def __init__(self, p, L):
#             self.p = p
#             self.L = L

#         def __call__(self, X, Y=None, eval_gradient=False):
#             d = np.subtract.outer(X, Y if Y is not None else X)
#             f = np.exp(-2 * np.sin(np.pi / self.p * d)**2 / self.L**2)
#             if eval_gradient is False:
#                 return f
#             else:
#                 s = np.sin(d * np.pi / self.p)
#                 c = np.cos(d * np.pi / self.p)
#                 j1 = 2.0 * np.pi * d * 2 * s * c * f / self.p**2 / self.L**2
#                 j2 = 4.0 * s**2 * f / self.L**3
#                 return f, np.stack((j1, j2), axis=2)

#         def diag(self, X):
#             return np.ones_like(X)

#         @property
#         def theta(self):
#             return np.log([self.p, self.L])

#         @theta.setter
#         def theta(self, t):
#             self.p, self.L = np.exp(t)

#         @property
#         def bounds(self):
#             return np.log([[1e-2, 10], [1e-2, 10]])

#     X = np.linspace(0, 1, 16, endpoint=False)
#     y = np.sin(X * 4 * np.pi)
#     kernel = Kernel(0.49, 0.1)
#     gpr = GPRNoiseDetector(kernel=kernel, alpha=1e-10, optimizer=True)
#     gpr.fit(X, y, tol=1e-5, repeat=repeat, verbose=verbose)
#     assert(kernel.p == pytest.approx(0.5, 1e-2))


def test_gpr_log_marginal_likelihood():

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

    X = np.linspace(-1, 1, 6, endpoint=False)
    y = np.sin(X * np.pi)
    eps = 1e-4
    for L in np.logspace(-1, 0.5, 10):
        kernel = Kernel(L)
        gpr = GPRNoiseDetector(kernel=kernel, alpha_bounds=(1e-8, 1.0))

        theta_ext0 = np.concatenate((
            np.copy(kernel.theta),
            np.log(0.01 * np.ones_like(y))
        ))

        _, dL = gpr.log_marginal_likelihood(
            theta_ext=theta_ext0, X=X, y=y, eval_gradient=True)

        for i in range(len(theta_ext0)):
            theta_ext_neg = np.copy(theta_ext0)
            theta_ext_pos = np.copy(theta_ext0)
            theta_ext_neg[i] -= eps
            theta_ext_pos[i] += eps
            L_pos = gpr.log_marginal_likelihood(
                theta_ext=theta_ext_pos, X=X, y=y, eval_gradient=False
            ).item()
            L_neg = gpr.log_marginal_likelihood(
                theta_ext=theta_ext_neg, X=X, y=y, eval_gradient=False
            ).item()

            dL_diff = (L_pos - L_neg) / (2 * eps)
            assert(dL[i] == pytest.approx(dL_diff, 1e-3, 1e-3))
