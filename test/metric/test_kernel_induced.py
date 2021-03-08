#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import sympy as sym
import pytest
from graphdot.util.pretty_tuple import pretty_tuple
from graphdot.metric import KernelInducedDistance


class Kernel:
    def __init__(self, v, L):
        self.v = v
        self.L = L
        self.expr = sym.sympify('v * exp(-1/2 * d^2 / L^2)')
        self.expr_jac = [
            sym.diff(self.expr, s) for s in ['v', 'L']
        ]
        self.expr_hess = [
            [sym.diff(sym.diff(self.expr, s1), s2)
             for s1 in ['v', 'L']]
            for s2 in ['v', 'L']
        ]

    def __call__(self, X, Y=None, eval_gradient=False):
        fun = sym.lambdify('d', self.expr.subs([('v', self.v), ('L', self.L)]))
        jac = [sym.lambdify('d', j.subs([('v', self.v), ('L', self.L)]))
               for j in self.expr_jac]
        d = np.subtract.outer(X, Y if Y is not None else X)
        f = fun(d)
        if eval_gradient is False:
            return f
        else:
            grad = np.empty((*f.shape, len(jac)), order='F')
            for i, j in enumerate(jac):
                grad[:, :, i] = j(d)
            return f, grad

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

    @property
    def hyperparameters(self):
        return pretty_tuple('Kernel', ['v', 'L'])(self.v, self.L)

    def clone_with_theta(self, theta=None):
        if theta is None:
            theta = self.theta
        return Kernel(*np.exp(theta))


def test_basic():
    d = KernelInducedDistance(kernel=Kernel(1.0, 1.0))
    X = np.arange(4)
    Y = np.arange(2)

    distance = d(X)
    assert d.hyperparameters is not None
    assert distance.shape == (len(X), len(X))
    assert np.allclose(distance.diagonal(), 0, atol=1e-3)
    assert np.all(distance >= 0)
    assert np.allclose(distance, distance.T, rtol=1e-14, atol=1e-14)
    distance, gradient = d(X, eval_gradient=True)
    assert gradient.shape == (len(X), len(X), len(d.theta))

    distance = d(X, X)
    assert distance.shape == (len(X), len(X))
    assert np.allclose(distance.diagonal(), 0, atol=1e-3)
    assert np.all(distance >= 0)
    assert np.allclose(distance, distance.T, rtol=1e-4, atol=1e-4)

    distance = d(X, Y)
    assert distance.shape == (len(X), len(Y))
    assert np.all(distance >= 0)


@pytest.mark.parametrize('X', [
    np.linspace(-1, 1, 4),
    np.linspace(-1, 1, 40),
    np.linspace(-10, 10, 40),
    np.random.randn(10) * 3.0,
    np.random.rand(10) * 3.0,
])
@pytest.mark.parametrize('kernel', [
    Kernel(1.5, 0.5),
    Kernel(1.5, 1.5),
    Kernel(0.2, 0.5),
])
def test_gradient(X, kernel):
    d = KernelInducedDistance(kernel=kernel)

    D, G = d(X, eval_gradient=True)
    delta = 1e-2
    for i, _ in enumerate(d.theta):
        h_pos, h_neg = np.exp(d.theta), np.exp(d.theta)
        h_pos[i] += delta
        h_neg[i] -= delta
        pos = d.clone_with_theta(np.log(h_pos))
        neg = d.clone_with_theta(np.log(h_neg))
        diff = (pos(X) - neg(X)) / (2 * delta)
        assert np.allclose(G[:, :, i], diff, rtol=1e-3, atol=1e-3)
