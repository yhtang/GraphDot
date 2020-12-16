#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest.mock
import pytest
import numpy as np
import os
import tempfile
from graphdot.model.gaussian_process.base import GaussianProcessRegressorBase

np.random.seed(0)


@pytest.mark.parametrize('normalize_y', [False, True])
def test_gpr_store_X_y(normalize_y):
    gpr = GaussianProcessRegressorBase(
        kernel=None,
        normalize_y=normalize_y,
        regularization='+',
        kernel_options={}
    )
    with pytest.raises(AttributeError):
        gpr.X
    with pytest.raises(AttributeError):
        gpr.y
    X = [1, 2, 3]
    y = [4, 5, 6]
    gpr.X = X
    gpr.y = y
    assert(isinstance(gpr.X, np.ndarray))
    assert(isinstance(gpr.y, np.ndarray))
    assert(gpr.X == pytest.approx(np.array(X)))
    if normalize_y:
        assert(gpr.y == pytest.approx(np.array(y)))
        assert(np.mean(gpr._y) == pytest.approx(0))
        assert(np.std(gpr._y) == pytest.approx(1))
        assert(gpr._ymean == pytest.approx(np.mean(y)))
        assert(gpr._ystd == pytest.approx(np.std(y)))
    else:
        assert(gpr.y == pytest.approx(np.array(y)))
        assert(gpr._y == pytest.approx(np.array(y)))
        assert(gpr._ymean == 0)
        assert(gpr._ystd == 1)


def test_gpr_gramian():
    class Kernel:
        def __call__(self, X, Y=None, eval_gradient=False):
            d = np.subtract.outer(X, Y if Y is not None else X)
            f = np.exp(-2 * np.sin(np.pi * d)**2)
            if eval_gradient is False:
                return f
            else:
                s = np.sin(d * np.pi)
                c = np.cos(d * np.pi)
                j1 = 2.0 * np.pi * d * 2 * s * c * f
                j2 = 4.0 * s**2 * f
                return f, np.stack((j1, j2), axis=2)

        def diag(self, X):
            return np.ones_like(X)

    gpr = GaussianProcessRegressorBase(
        kernel=Kernel(),
        normalize_y=False,
        regularization='+',
        kernel_options={}
    )

    n = 5
    m = 3
    X = np.ones(n)
    Y = np.ones(m)

    assert(gpr._gramian(0, X) == pytest.approx(np.ones((n, n))))
    assert(gpr._gramian(0, Y) == pytest.approx(np.ones((m, m))))
    assert(gpr._gramian(0, X, Y) == pytest.approx(np.ones((n, m))))
    assert(gpr._gramian(0, Y, X) == pytest.approx(np.ones((m, n))))
    assert(len(gpr._gramian(0, X, jac=True)) == 2)
    assert(gpr._gramian(0, X, jac=True)[1].shape == (n, n, 2))
    assert(gpr._gramian(0, Y, jac=True)[1].shape == (m, m, 2))
    assert(len(gpr._gramian(0, X, Y, jac=True)) == 2)
    assert(gpr._gramian(0, X, Y, jac=True)[1].shape == (n, m, 2))
    assert(gpr._gramian(0, X, diag=True) == pytest.approx(np.ones(n)))
    assert(gpr._gramian(0, Y, diag=True) == pytest.approx(np.ones(m)))
    with pytest.raises(ValueError):
        gpr._gramian(0, X, Y, diag=True)


def test_kernel_options():
    n = 3
    kernel = unittest.mock.MagicMock()
    kernel.return_value = np.eye(n)
    kernel.diag.return_value = np.ones(n)

    options = {
        'a': 1,
        'b': False
    }
    gpr = GaussianProcessRegressorBase(
        kernel=kernel,
        normalize_y=False,
        regularization='+',
        kernel_options=options
    )
    X = np.ones(n)
    Y = np.zeros(n)

    gpr._gramian(0, X)
    kernel.assert_called_with(X, **options)

    gpr._gramian(0, Y, X)
    kernel.assert_called_with(Y, X, **options)

    gpr._gramian(0, Y)
    kernel.assert_called_with(Y, **options)


@pytest.mark.parametrize('normalize_y', [True, False])
def test_gpr_save_load(normalize_y):

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

        @property
        def bounds(self):
            return np.log([[1e-2, 10]])

        def clone_with_theta(self, theta):
            k = Kernel(1.0)
            k.theta = theta
            return k

    X = np.linspace(-1, 1, 6, endpoint=False)
    y = np.sin(X * np.pi)
    kernel = Kernel(0.1)
    gpr = GaussianProcessRegressorBase(
        kernel=kernel,
        normalize_y=normalize_y,
        regularization='+',
        kernel_options={}
    )
    gpr.X = X
    gpr.y = y

    with tempfile.TemporaryDirectory() as cwd:
        file = 'test-model.pkl'
        target_path = os.path.join(cwd, file)
        gpr.save(cwd, file)
        assert(os.path.exists(target_path))

        gpr_saved = GaussianProcessRegressorBase(
            kernel=kernel,
            normalize_y=normalize_y,
            regularization='+',
            kernel_options={}
        )
        gpr_saved.load(cwd, file)

        assert(gpr._ymean == pytest.approx(gpr_saved._ymean))
        assert(gpr._ystd == pytest.approx(gpr_saved._ystd))
        assert(gpr._y == pytest.approx(gpr_saved._y))
