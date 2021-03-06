#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest
import numpy as np
from graphdot.model.gaussian_process import GaussianProcessRegressor

np.random.seed(0)


def test_gpr_singular_kernel_matrix():

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

    X = np.ones(3)
    y = np.random.rand(3)
    gpr = GaussianProcessRegressor(kernel=Kernel(1.0), alpha=0)
    gpr.fit(X, y)
    z = gpr.predict(X)
    assert(z == pytest.approx(np.mean(y)))


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


def test_gpr_fit_masked_target():

    class Kernel:
        def __call__(self, X, Y=None):
            return np.exp(-np.subtract.outer(X, Y if Y is not None else X)**2)

        def diag(self, X):
            return np.ones_like(X)

    X = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    y = np.random.randn(10)
    bad = [1, 4, 7]
    y[bad] = None
    kernel = Kernel()
    gpr = GaussianProcessRegressor(kernel=kernel, alpha=1e-12)
    gpr.fit(X, y)
    assert(np.all(np.isfinite(gpr.predict(X))))

    baseline = GaussianProcessRegressor(kernel=kernel, alpha=1e-12)
    baseline.fit(X[~np.isnan(y)], y[~np.isnan(y)])
    grid = np.linspace(-1, 10, 100)
    assert(np.allclose(gpr.predict(grid), baseline.predict(grid)))


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


@pytest.mark.parametrize('f', [
    lambda x: np.cos(x * np.pi),
    lambda x: np.sin(x * np.pi) + 0.5 * x + 1.0 * x**2
])
def test_gpr_predict_loocv(f):

    class Kernel:
        def __call__(self, X, Y=None):
            return np.exp(-np.subtract.outer(X, Y if Y is not None else X)**2)

        def diag(self, X):
            return np.ones_like(X)

    kernel = Kernel()
    gpr = GaussianProcessRegressor(kernel=kernel, alpha=1e-12)
    X = np.linspace(-1, 1, 6)
    y = f(X)
    y_loocv, std_loocv = gpr.predict_loocv(X, y, return_std=True)
    assert(y_loocv == pytest.approx(gpr.predict_loocv(X, y, return_std=False)))
    for i, _ in enumerate(X):
        Xi = np.delete(X, i)
        yi = np.delete(y, i)
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


@pytest.mark.parametrize('f', [
    lambda x: np.cos(x * np.pi),
    lambda x: np.sin(x * np.pi) + 0.5 * x + 1.0 * x**2
])
def test_gpr_squared_loocv_error(f):

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
    y = f(X)
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
        assert(de.item() == pytest.approx(de_diff, 1e-2, 1e-2))


@pytest.mark.parametrize('normalize_y', [True, False])
def test_gpr_fit_loocv_no_opt(normalize_y):

    class Kernel:
        def __call__(self, X, Y=None):
            d = np.subtract.outer(X, Y if Y is not None else X)
            return np.exp(-2 * np.sin(np.pi / 0.5 * d)**2)

        def diag(self, X):
            return np.ones_like(X)

    X = np.linspace(0, 1, 16, endpoint=False)
    y = np.sin(X * 4 * np.pi)
    mask = np.array([1, 0, 1, 0, 1, 0, 1, 0,
                     0, 1, 0, 1, 0, 1, 0, 1], dtype=np.bool_)
    kernel = Kernel()
    gpr = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-10,
        normalize_y=normalize_y,
        optimizer=False,
    )
    gpr.fit_loocv(X[mask], y[mask])
    z1 = gpr.predict(X[~mask])
    assert(z1 == pytest.approx(y[~mask], 1e-6))


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


@pytest.mark.parametrize('loss', ['likelihood', 'loocv'])
def test_gpr_fit_duplicate_x(loss):
    '''Training in the precense of duplicate input values.'''

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
            return np.log([[0.001, 10.0]])

        def clone_with_theta(self, theta):
            k = Kernel(1.0)
            k.theta = theta
            return k

    X = np.array([0, 1, 1, 2, 3, 3.995, 4, 6, 7, 8, 8.0001, 9])
    y = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1])
    kernel = Kernel(1.0)
    gpr = GaussianProcessRegressor(kernel=kernel, alpha=0, optimizer=True)
    gpr.fit(X, y, tol=1e-5, loss=loss)
    assert gpr.predict([1]) == pytest.approx(0.5)
    assert gpr.predict([4]) == pytest.approx(1.0)
    assert gpr.predict([8]) == pytest.approx(0.0)


def test_gpr_regularization():
    '''Training in the precense of duplicate input values.'''

    class Kernel:
        def __init__(self, v, L):
            self.v = v
            self.L = L

        def __call__(self, X, Y=None, eval_gradient=False):
            v = self.v
            L = self.L
            d = np.subtract.outer(X, Y if Y is not None else X)
            f = v**2 * np.exp(-0.5 * d**2 / L**2)
            if eval_gradient is False:
                return f
            else:
                j1 = v**2 * np.exp(-0.5 * d**2 / L**2) * d**2 * L**-3
                j2 = 2 * v * f
                return f, np.stack((j1, j2), axis=2)

        def diag(self, X):
            return np.ones_like(X)

        @property
        def theta(self):
            return np.log([self.v, self.L])

        @theta.setter
        def theta(self, t):
            self.v, self.L = np.exp(t[:2])

        @property
        def bounds(self):
            return np.log([[0.001, 10.0], [0.001, 10.0]])

        def clone_with_theta(self, theta):
            k = Kernel(1.0, 1.0)
            k.theta = theta
            return k

    X = np.array([0, 1, 1, 2])
    y = np.array([1, 0, 1, 0])
    gpr1 = GaussianProcessRegressor(
        kernel=Kernel(100.0, 1.0),
        alpha=1e-6,
        optimizer=False,
        regularization='*'
    )
    gpr2 = GaussianProcessRegressor(
        kernel=Kernel(100.0, 1.0),
        alpha=1e-4,
        optimizer=False,
        regularization='+'
    )
    gpr1.fit(X, y, tol=1e-5)
    gpr2.fit(X, y, tol=1e-5)
    grid = np.linspace(0, 2, 9)
    assert np.allclose(
        gpr1.predict(grid), gpr2.predict(grid), rtol=1e-5, atol=1e-5
    )
