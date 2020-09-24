#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest.mock
import pytest
import numpy as np
import os
import tempfile
from graphdot.model.gaussian_process import GPROutlierDetector

np.random.seed(0)


class RBFKernel:
    def __init__(self, v, L):
        self.v = v
        self.L = L

    def __call__(self, X, Y=None, eval_gradient=False):
        v, L = self.v, self.L
        d = np.subtract.outer(X, Y if Y is not None else X)
        f = v * np.exp(-0.5 * d**2 / L**2)
        if eval_gradient is False:
            return f
        else:
            j1 = np.exp(-0.5 * d**2 / L**2)
            j2 = v * np.exp(-0.5 * d**2 / L**2) * d**2 * L**-3
            return f, np.stack((j1, j2), axis=2)

    def diag(self, X):
        return np.ones_like(X)

    @property
    def theta(self):
        return np.log([self.v, self.L])

    @theta.setter
    def theta(self, t):
        self.v, self.L = np.exp(t)

    @property
    def bounds(self):
        return np.log([[1e-5, 1e5], [1e-2, 10]])

    def clone_with_theta(self, theta):
        k = RBFKernel(1.0, 1.0)
        k.theta = theta
        return k


@pytest.mark.parametrize('v', [0.25, 0.5, 1.0, 2.0, 5.0])
@pytest.mark.parametrize('L', np.logspace(-1, 0.5, 10))
def test_gpr_log_marginal_likelihood(v, L):

    eps = 1e-4
    X = np.linspace(-1, 1, 6, endpoint=False)
    y = np.sin(X * np.pi)

    kernel = RBFKernel(v, L)
    gpr = GPROutlierDetector(kernel=kernel)

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


# @pytest.mark.parametrize('repeat', [3, 5])
# @pytest.mark.parametrize('verbose', [True, False])
# def test_gpr_fit_mle(repeat, verbose):

#     X = np.linspace(0, 1, 16, endpoint=False)
#     y = np.sin(X * 4 * np.pi)
#     kernel = Kernel(0.49, 0.1)
#     gpr = GPROutlierDetector(kernel=kernel, optimizer=True)
#     gpr.fit(X, y, w=0.1, tol=1e-5, repeat=repeat, verbose=verbose)
#     assert(kernel.p == pytest.approx(0.5, 1e-2))
#     assert(gpr.y_uncertainty == pytest.approx(0, abs=1e-2))


