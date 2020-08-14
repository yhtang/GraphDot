#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest
import numpy as np
from graphdot.kernel.fix import Normalization


def test_normalization():

    class DotProductKernel:
        def __init__(self, xi):
            self.xi = xi

        def __call__(self, X, Y=None, eval_gradient=False):
            if Y is None:
                k = X @ X.T
            else:
                k = X @ Y.T
            K = k**self.xi
            if eval_gradient is False:
                return K
            else:
                j = self.xi * k**(self.xi - 1)
                return K, np.stack((j, ), axis=2)

        def diag(self, X, eval_gradient=False):
            k = np.sum(X**2, axis=1)
            K = k**self.xi
            if eval_gradient is True:
                return K, np.array([self.xi * k**(self.xi - 1)])
            else:
                return K

        @property
        def theta(self):
            return np.log([self.xi])

        @theta.setter
        def theta(self, t):
            self.xi = np.exp(t[0])

        @property
        def bounds(self):
            return np.log([[1e-2, 10]])

        def clone_with_theta(self, theta):
            k = DotProductKernel(1.0)
            k.theta = theta
            return k

    kernel = Normalization(DotProductKernel(2.0))

    nX, nY, d = 20, 10, 3
    X = np.random.rand(nX, d)
    Y = np.random.rand(nY, d)

    K, dK = kernel(X, eval_gradient=True)
    assert(K.shape == (nX, nX))
    assert(dK.shape == (nX, nX, len(kernel.theta)))
    assert(dK.flags['F_CONTIGUOUS'])
    assert(K.diagonal() == pytest.approx(1.0, 1e-8))

    K, dK = kernel(X, Y, eval_gradient=True)
    assert(K.shape == (nX, nY))
    assert(dK.shape == (nX, nY, len(kernel.theta)))
    assert(dK.flags['F_CONTIGUOUS'])
