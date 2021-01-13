#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pytest
from scipy.spatial import distance_matrix as pairwise_distances
from scipy.spatial.distance import cdist
from unittest.mock import MagicMock
from graphdot.model.gaussian_field import GaussianFieldRegressor
from graphdot.model.gaussian_field import Weight


@pytest.mark.parametrize('case', [
    # [A] --1.0-- [B] --1.0-- [C]
    (
        np.array([
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ]),
        [True, False, True],
        lambda y, z: z[1] == pytest.approx(0.5 * (y[0] + y[2]), abs=1e-4)
    ),
    # [A] --3.0-- [B] --1.0-- [C]
    (
        np.array([
            [0.0, 3.0, 0.0],
            [3.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ]),
        [True, False, True],
        lambda y, z: z[1] == pytest.approx(0.75 * y[0] + 0.25 * y[2], abs=1e-4)
    ),
    # [A] --1.0-- [B] --1.0-- [C] --1.0-- [D]
    (
        np.array([
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
        ]),
        [True, False, False, True],
        lambda y, z: (
            z[1] == pytest.approx((2 * y[0] + 1 * y[-1]) / 3, abs=1e-4) and
            z[2] == pytest.approx((1 * y[0] + 2 * y[-1]) / 3, abs=1e-4)
        )
    ),
    # Fully connected
    (
        np.array([
            [0.0, 1.0, 1.0, 1.0],
            [1.0, 0.0, 1.0, 1.0],
            [1.0, 1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0, 0.0],
        ]),
        [True, True, True, False],
        lambda y, z: (
            z[3] == pytest.approx((y[0] + y[1] + y[2]) / 3, abs=1e-4)
        )
    ),
])
def test_gaussian_field_prediction(case):
    W, labeled, verify = case

    class WeightLookUpTable:
        def __call__(self, X, Y=None):
            if Y is None:
                return W[X, :][:, X]
            else:
                return W[X, :][:, Y]

    g = GaussianFieldRegressor(WeightLookUpTable(), smoothing=0)

    for _ in range(100):
        X = np.arange(len(W))
        y = np.random.randn(len(W))
        y[~np.array(labeled)] = np.nan
        z = g.fit_predict(X, y)
        assert len(y) == len(z)
        assert verify(y, z)


@pytest.mark.parametrize('case', [
    # [A] --1.0-- [B] --1.0-- [C]
    (
        np.array([
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ]),
        [True, False, True],
        [[0.5, 0.5]],
    ),
    # [A] --3.0-- [B] --1.0-- [C]
    (
        np.array([
            [0.0, 3.0, 0.0],
            [3.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ]),
        [True, False, True],
        [[0.75, 0.25]]
    ),
    # [A] --1.0-- [B] --1.0-- [C] --1.0-- [D]
    (
        np.array([
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
        ]),
        [True, False, False, True],
        [[2/3, 1/3],
         [1/3, 2/3]]
    ),
    # Fully connected
    (
        np.array([
            [0.0, 1.0, 1.0, 1.0],
            [1.0, 0.0, 1.0, 1.0],
            [1.0, 1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0, 0.0],
        ]),
        [True, True, True, False],
        [[1/3, 1/3, 1/3]]
    ),
])
def test_gaussian_field_influence(case):
    W, labeled, truth = case

    class WeightLookUpTable:
        def __call__(self, X, Y=None):
            if Y is None:
                return W[X, :][:, X]
            else:
                return W[X, :][:, Y]

    g = GaussianFieldRegressor(WeightLookUpTable(), smoothing=0)

    X = np.arange(len(W))
    y = np.random.randn(len(W))
    y[~np.array(labeled)] = np.nan
    z, influence = g.fit_predict(X, y, return_influence=True)
    assert np.allclose(influence, truth)


def test_average_label_entropy():

    g = GaussianFieldRegressor(weight='precomputed', smoothing=0)

    e = g.average_label_entropy(
        X=np.array([
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ]),
        y=np.array([0, np.nan, 1])
    )

    assert e == pytest.approx(-np.log(0.5))


@pytest.mark.parametrize('n', [4, 7, 9, 16, 25])
@pytest.mark.parametrize('k', [2, 3, 5, 8])
@pytest.mark.parametrize('d', [1, 2, 4, 7, 20])
@pytest.mark.parametrize('smoothing', [0, 0.1, 0.5])
def test_average_label_entropy_gradient(n, k, d, smoothing):

    class OneOverRn:
        '''
        w = 1 / (r + a)^b
        '''
        def __init__(self, a=0.1, b=1):
            self.a = a
            self.b = b

        def __call__(self, X, Y=None, eval_gradient=False):
            '''
            Parameters
            ----------
            eval_gradient: bool
                If true, also return the gradient of the weights with respect to
                the **log-scale** hyperparameters.
            '''
            d = self.a + (cdist(X, X) if Y is None else cdist(X, Y))
            w = d**-self.b
            j1 = -self.b * d**(-self.b - 1)
            j2 = -d**(-self.b) * np.log(d)
            if eval_gradient:
                return w, np.stack(
                    [j1, j2], axis=2
                ) * np.exp(self.theta)[None, None, :]
            else:
                return w

        @property
        def theta(self):
            return np.log([self.a, self.b])

        @theta.setter
        def theta(self, values):
            self.a, self.b = np.exp(values)

        @property
        def bounds(self):
            return np.log([
                [0.001, 100.0],
                [0.001, 100.0]
            ])

    gfr = GaussianFieldRegressor(
        weight=OneOverRn(a=1.0, b=1.0),
        smoothing=smoothing
    )
    X = np.random.randn(n, d)
    y = np.random.rand(n)
    y[np.random.choice(n, max(1, n // k), replace=False)] = np.nan

    _, dloss = gfr.average_label_entropy(X, y, eval_gradient=True)

    eps = 1e-3
    theta = np.copy(gfr.weight.theta)
    for i in range(len(theta)):
        pos, neg = theta.copy(), theta.copy()
        pos[i] += eps
        neg[i] -= eps
        f_pos = gfr.average_label_entropy(X, y, theta=pos)
        f_neg = gfr.average_label_entropy(X, y, theta=neg)
        delta = (f_pos - f_neg) / (2 * eps)
        assert delta == pytest.approx(dloss[i], rel=1e-5, abs=1e-6)


def test_loocv_error():

    g = GaussianFieldRegressor(weight='precomputed', smoothing=0)

    assert g.loocv_error(
        X=np.array([
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ]),
        y=np.zeros(3)
    ) == pytest.approx(0)

    assert g.loocv_error(
        X=np.array([
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ]),
        y=np.ones(3)
    ) == pytest.approx(0)

    assert g.loocv_error(
        X=np.array([
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ]),
        y=-np.ones(3)
    ) == pytest.approx(0)

    assert g.loocv_error(
        X=np.array([
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ]),
        y=np.array([-1.0, 0.0, 1.0]),
        p=1
    ) == pytest.approx(1.0)

    assert g.loocv_error(
        X=np.array([
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ]),
        y=np.array([-1.0, 0.0, 1.0]),
        p=2
    ) == pytest.approx(np.sqrt(1.5))


@pytest.mark.parametrize('n', [4, 7, 25])
@pytest.mark.parametrize('k', [2, 3, 8])
@pytest.mark.parametrize('d', [1, 2, 4, 20])
@pytest.mark.parametrize('p', [1, 1.5, 2, 3])
@pytest.mark.parametrize('smoothing', [0, 0.1, 0.5])
def test_loocv_error_gradient(n, k, d, p, smoothing):

    class OneOverRn:
        '''
        w = 1 / (r + a)^b
        '''
        def __init__(self, a=0.1, b=1):
            self.a = a
            self.b = b

        def __call__(self, X, Y=None, eval_gradient=False):
            '''
            Parameters
            ----------
            eval_gradient: bool
                If true, also return the gradient of the weights with respect to
                the **log-scale** hyperparameters.
            '''
            d = self.a + (cdist(X, X) if Y is None else cdist(X, Y))
            w = d**-self.b
            j1 = -self.b * d**(-self.b - 1)
            j2 = -d**(-self.b) * np.log(d)
            if eval_gradient:
                return w, np.stack(
                    [j1, j2], axis=2
                ) * np.exp(self.theta)[None, None, :]
            else:
                return w

        @property
        def theta(self):
            return np.log([self.a, self.b])

        @theta.setter
        def theta(self, values):
            self.a, self.b = np.exp(values)

        @property
        def bounds(self):
            return np.log([
                [0.001, 100.0],
                [0.001, 100.0]
            ])

    gfr = GaussianFieldRegressor(
        weight=OneOverRn(a=1.0, b=1.0),
        smoothing=smoothing
    )
    X = np.random.randn(n, d)
    y = np.random.rand(n)
    y[np.random.choice(n, max(1, n // k), replace=False)] = np.nan

    _, dloss = gfr.loocv_error(X, y, p=p, eval_gradient=True)

    eps = 1e-3
    theta = np.copy(gfr.weight.theta)
    for i in range(len(theta)):
        pos, neg = theta.copy(), theta.copy()
        pos[i] += eps
        neg[i] -= eps
        f_pos = gfr.loocv_error(X, y, p=p, theta=pos)
        f_neg = gfr.loocv_error(X, y, p=p, theta=neg)
        delta = (f_pos - f_neg) / (2 * eps)
        assert delta == pytest.approx(dloss[i], rel=1e-5, abs=1e-6)
