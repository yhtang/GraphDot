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


def test_gpr_check_matrix():

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

    with pytest.raises(np.linalg.LinAlgError):
        X = np.ones(3)
        y = np.ones(3)
        gpr = GaussianProcessRegressor(kernel=Kernel(1.0), alpha=0)
        gpr.fit(X, y)


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
    with pytest.raises(RuntimeError):
        gpr.predict(X)
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
    assert(y_loocv == pytest.approx(gpr.predict_loocv(X, y, return_std=False)))
    for i, _ in enumerate(X):
        Xi = np.concatenate((X[:i], X[i+1:]))
        yi = np.concatenate((y[:i], y[i+1:]))
        gpr_loocv = GaussianProcessRegressor(kernel=kernel, alpha=1e-12)
        gpr_loocv.fit(Xi, yi)
        y_loocv_i, std_loocv_i = gpr_loocv.predict(X[[i]], return_std=True)
        assert(y_loocv_i.item() == pytest.approx(y_loocv[i], 1e-7))
        assert(std_loocv_i.item() == pytest.approx(std_loocv[i], 1e-7))


@pytest.mark.parametrize('repeat', [1, 3])
@pytest.mark.parametrize('verbose', [True, False])
def test_gpr_fit_mle(repeat, verbose):
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
    gpr.fit(X, y, tol=1e-5, repeat=repeat, verbose=verbose)
    assert(kernel.p == pytest.approx(0.5, 1e-2))


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
        gpr = GaussianProcessRegressor(kernel=kernel, alpha=1e-10)
        _, dL = gpr.log_marginal_likelihood(X=X, y=y, eval_gradient=True)
        theta0 = np.copy(kernel.theta)

        L_pos = gpr.log_marginal_likelihood(
            theta=theta0 + eps, X=X, y=y, eval_gradient=False
        ).item()
        L_neg = gpr.log_marginal_likelihood(
            theta=theta0 - eps, X=X, y=y, eval_gradient=False
        ).item()

        dL_diff = (L_pos - L_neg) / (2 * eps)
        assert(dL.item() == pytest.approx(dL_diff, 1e-3, 1e-3))


def test_gpr_squared_loocv_error():

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
    for L in np.logspace(-2, 1, 20):
        kernel = Kernel(L)
        gpr = GaussianProcessRegressor(kernel=kernel, alpha=1e-10)
        _, de = gpr.squared_loocv_error(X=X, y=y, eval_gradient=True)
        theta0 = np.copy(kernel.theta)

        e_pos = gpr.squared_loocv_error(
            theta=theta0 + eps, X=X, y=y, eval_gradient=False
        ).item()
        e_neg = gpr.squared_loocv_error(
            theta=theta0 - eps, X=X, y=y, eval_gradient=False
        ).item()

        de_diff = (e_pos - e_neg) / (2 * eps)
        assert(de.item() == pytest.approx(de_diff, 1e-3, 1e-3))


@pytest.mark.parametrize('normalize_y', [True, False])
def test_gpr_fit_loocv_no_opt(normalize_y):

    class Kernel:
        def __init__(self, L):
            self.L = L

        def __call__(self, X, Y=None):
            L = self.L
            d = np.subtract.outer(X, Y if Y is not None else X)
            return np.exp(-0.5 * d**2 / L**2)

        def diag(self, X):
            return np.ones_like(X)

    X = np.linspace(-1, 1, 6, endpoint=False)
    y = np.sin(X * np.pi)
    kernel = Kernel(0.1)
    gpr = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-10,
        normalize_y=normalize_y,
        optimizer=False,
    )
    _, m1, s1 = gpr.fit_loocv(X, y, return_mean=True, return_std=True)
    m2, s2 = gpr.predict_loocv(X, y, return_std=True)
    assert(m1 == pytest.approx(m2))
    assert(s1 == pytest.approx(s2))


@pytest.mark.parametrize('normalize_y', [True, False])
@pytest.mark.parametrize('repeat', [1, 3])
def test_gpr_fit_loocv_opt(normalize_y, repeat):

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
    gpr = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-10,
        normalize_y=normalize_y,
        optimizer=True
    )
    e1 = gpr.squared_loocv_error(X=X, y=y)
    gpr.fit_loocv(X, y, repeat=repeat, verbose=True)
    e2 = gpr.squared_loocv_error(X=X, y=y)
    assert(e1 > e2)
    assert(kernel.L == pytest.approx(0.86, 0.01))


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
