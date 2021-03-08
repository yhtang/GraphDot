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
        jac = [sym.lambdify('d', j.subs([('v', self.v), ('L', self.L)])) for j in self.expr_jac]
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


# def test_maximin_basic():
#     metric = MaxiMin(
#         node_kernel=TensorProduct(
#             element=KroneckerDelta(0.5)
#         ),
#         edge_kernel=TensorProduct(
#             length=SquareExponential(0.1)
#         ),
#         q=0.01
#     )
#     distance = metric(G)
#     assert distance.shape == (len(G), len(G))
#     assert np.allclose(distance.diagonal(), 0, atol=1e-3)
#     assert np.all(distance >= 0)
#     assert np.allclose(distance, distance.T, rtol=1e-14, atol=1e-14)

#     distance = metric(G, G)
#     assert distance.shape == (len(G), len(G))
#     assert np.allclose(distance.diagonal(), 0, atol=1e-3)
#     assert np.all(distance >= 0)
#     assert np.allclose(distance, distance.T, rtol=1e-4, atol=1e-4)

#     distance = metric(G, H)
#     assert distance.shape == (len(G), len(H))
#     assert np.all(distance >= 0)


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
