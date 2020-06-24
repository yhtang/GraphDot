#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest.mock
import pytest
import numpy as np
from graphdot.model.gaussian_process import GaussianProcessRegressor

np.random.seed(0)


@pytest.mark.parametrize('normalize_y', [False, True])
def test_gpr_store_X_y(normalize_y):
    gpr = GaussianProcessRegressor(kernel=None, normalize_y=normalize_y)
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
        assert(np.mean(gpr.y) == pytest.approx(0))
        assert(np.std(gpr.y) == pytest.approx(1))
        assert(gpr.y_mean == pytest.approx(np.mean(y)))
        assert(gpr.y_std == pytest.approx(np.std(y)))
    else:
        assert(gpr.y == pytest.approx(np.array(y)))
        assert(gpr.y_mean == 0)
        assert(gpr.y_std == 1)


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

    gpr = GaussianProcessRegressor(kernel=Kernel())

    n = 5
    m = 3
    X = np.ones(n)
    Y = np.ones(m)

    assert(gpr._gramian(X) == pytest.approx(np.ones((n, n))))
    assert(gpr._gramian(Y) == pytest.approx(np.ones((m, m))))
    assert(gpr._gramian(X, Y) == pytest.approx(np.ones((n, m))))
    assert(gpr._gramian(Y, X) == pytest.approx(np.ones((m, n))))
    assert(len(gpr._gramian(X, jac=True)) == 2)
    assert(gpr._gramian(X, jac=True)[1].shape == (n, n, 2))
    assert(gpr._gramian(Y, jac=True)[1].shape == (m, m, 2))
    assert(len(gpr._gramian(X, Y, jac=True)) == 2)
    assert(gpr._gramian(X, Y, jac=True)[1].shape == (n, m, 2))
    assert(gpr._gramian(X, diag=True) == pytest.approx(np.ones(n)))
    assert(gpr._gramian(Y, diag=True) == pytest.approx(np.ones(m)))
    with pytest.raises(ValueError):
        gpr._gramian(X, Y, diag=True)


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
    z = gpr.predict(X)
    assert(z == pytest.approx(y, 1e-3, 1e-3))
    z, std = gpr.predict(X, return_std=True)
    assert(z == pytest.approx(y, 1e-3, 1e-3))
    assert(std == pytest.approx(np.zeros_like(y), 1e-3, 1e-3))
    z, cov = gpr.predict(X, return_cov=True)
    assert(z == pytest.approx(y, 1e-3, 1e-3))
    assert(cov == pytest.approx(np.zeros((len(X), len(X))), 1e-3, 1e-3))


def test_gpr_predict_periodic():
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


def test_gpr_predict_loocv():

    class Kernel:
        def __call__(self, X, Y=None):
            return np.exp(-np.subtract.outer(X, Y if Y is not None else X)**2)

        def diag(self, X):
            return np.ones_like(X)

    kernel = Kernel()
    gpr = GaussianProcessRegressor(kernel=kernel, alpha=1e-12)
    X = np.linspace(-1, 1, 6)
    y = np.cos(X * np.pi)
    y_loocv, std_loocv = gpr.predict_loocv(X, y, return_std=True)
    for i, _ in enumerate(X):
        Xi = np.concatenate((X[:i], X[i+1:]))
        yi = np.concatenate((y[:i], y[i+1:]))
        gpr_loocv = GaussianProcessRegressor(kernel=kernel, alpha=1e-12)
        gpr_loocv.fit(Xi, yi)
        y_loocv_i, std_loocv_i = gpr_loocv.predict(X[[i]], return_std=True)
        assert(y_loocv_i.item() == pytest.approx(y_loocv[i], 1e-7))
        assert(std_loocv_i.item() == pytest.approx(std_loocv[i], 1e-7))


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


def test_kernel_options():
    n = 3
    kernel = unittest.mock.MagicMock()
    kernel.return_value = np.eye(n)
    kernel.diag.return_value = np.ones(n)

    options = {
        'a': 1,
        'b': False
    }
    gpr = GaussianProcessRegressor(kernel=kernel, kernel_options=options)
    X = np.ones(n)
    Y = np.zeros(n)
    y = np.ones(n)

    gpr.fit(X, y)
    kernel.assert_called_with(X, **options)

    gpr.predict(Y, return_std=True)
    kernel.assert_called_with(Y, X, **options)
    kernel.diag.assert_called_with(Y, **options)

    gpr.predict(Y, return_cov=True)
    kernel.assert_called_with(Y, **options)
