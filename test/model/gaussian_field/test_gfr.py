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
def test_average_label_entropy_gradient(n, k, d):

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
        smoothing=0
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
        assert delta == pytest.approx(dloss[i], rel=1e-5, abs=1e-8)



def test_laplacian():

    g = GaussianFieldRegressor(weight='precomputed', smoothing=0)

    e = g.laplacian(
        X=np.array([
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ]),
        y=np.zeros(3)
    )

    assert e == pytest.approx(0)


# def test_gaussian_field_prediction2():
#     """
#     Tests Gaussian Field Regressor Prediction
#     """
#     mock = MagicMock()
#     X = np.array([0])
#     y = np.array([1])
#     W_ul = np.array([
#             [1],
#             [1],
#             [1],
#             [1],
#         ])
#     W_uu = np.array([
#             [0, 1, 1, 1],
#             [1, 0, 1, 1],
#             [1, 1, 0, 1],
#             [1, 1, 1, 0],
#         ])
#     Xt = np.array([1, 2, 3, 4])
#     mock.side_effect = [W_ul, W_uu]
#     g = GaussianFieldRegressor(mock, smoothing=0)
#     g.set_model(X, y)
#     A = g.predict(Xt)
#     B = np.array([1, 1, 1, 1])
#     assert np.allclose(A, B)


# def test_smoothing():
#     """
#     Tests Gaussian Field Regressor Prediction
#     """
#     mock = MagicMock()
#     X = np.array([0, 1])
#     y = np.array([0, 1])
#     W_ul = np.array([
#             [1, 0],
#             [0, 0],
#             [0, 1],
#         ])
#     W_uu = np.array([
#             [0, 1, 0],
#             [1, 0, 1],
#             [0, 1, 0],
#         ])
#     Xt = np.array([0, 1, 2])
#     mock.side_effect = [W_ul, W_uu]
#     g = GaussianFieldRegressor(mock, smoothing=1e-2)
#     g.set_model(X, y)
#     A = g.predict(Xt)
#     B = np.array([.2525, .5, .7475])
#     assert np.allclose(A, B)


# def test_gaussian_field_influence_matrix():
#     """
#     Tests Gaussian Field Regressor Prediction
#     """
#     mock = MagicMock()
#     X = np.array([0])
#     y = np.array([1])
#     W_ul = np.array([
#             [1],
#             [1],
#             [1],
#             [1],
#         ])
#     W_uu = np.array([
#             [0, 1, 1, 1],
#             [1, 0, 1, 1],
#             [1, 1, 0, 1],
#             [1, 1, 1, 0],
#         ])
#     Xt = np.array([1, 2, 3, 4])
#     mock.side_effect = [W_ul, W_uu]
#     g = GaussianFieldRegressor(mock, smoothing=0)
#     g.set_model(X, y)
#     _, A, _ = g.predict(Xt, display=True)
#     B = np.array([1, 1, 1, 1])
#     assert np.allclose(np.dot(A, y), B)


# def test_squared_error_gradient():

#     class Test_Weight(Weight):
#         def __init__(self, sigma, bounds):
#             self.sigma = sigma
#             self._bounds = bounds

#         def __call__(self, X, Y=None, eval_gradient=False):
#             if Y is None:
#                 Y = X
#             m = pairwise_distances(X, Y)
#             w = np.exp(-(m/self.sigma[0])**2/2)
#             if eval_gradient:
#                 return w, np.array([m**2/self.sigma[0]**3]) * w
#             else:
#                 return w

#         @property
#         def theta(self):
#             return self.sigma

#         @theta.setter
#         def theta(self, values):
#             self.sigma = values

#         @property
#         def bounds(self):
#             return self._bounds

#     np.random.seed(0)
#     X_train = np.random.rand(20, 2)
#     y_train = np.random.rand(20)
#     X_test = np.random.rand(5, 2)
#     y_test = np.random.rand(5)

#     sigma = np.array([.5])
#     bounds = np.array([[.3, 1]])
#     eps = 1e-3
#     g = GaussianFieldRegressor(
#         Test_Weight(sigma, bounds),
#         optimizer=True,
#         smoothing=0,
#     )
#     g.set_model(X_train, y_train)
#     _, grad = g.squared_error(X_test, y_test, eval_gradient=True)

#     g.weight.theta += eps
#     f1 = g.squared_error(X_test, y_test)
#     g.weight.theta -= 2 * eps
#     f2 = g.squared_error(X_test, y_test)
#     g.weight.theta += eps
#     assert np.isclose((f1 - f2)/(2 * eps), grad[0])


# def test_cross_entropy_gradient():

#     class Test_Weight(Weight):
#         def __init__(self, sigma, bounds):
#             self.sigma = sigma
#             self._bounds = bounds

#         def __call__(self, X, Y=None, eval_gradient=False):
#             if Y is None:
#                 Y = X
#             m = pairwise_distances(X, Y)
#             w = np.exp(-(m/self.sigma[0])**2/2)
#             if eval_gradient:
#                 return w, np.array([m**2/self.sigma[0]**3]) * w
#             else:
#                 return w

#         @property
#         def theta(self):
#             return self.sigma

#         @theta.setter
#         def theta(self, values):
#             self.sigma = values

#         @property
#         def bounds(self):
#             return self._bounds

#     np.random.seed(0)
#     X_train = np.random.rand(20, 2)
#     y_train = np.random.rand(20)
#     X_test = np.random.rand(5, 2)
#     y_test = np.random.rand(5)

#     sigma = np.array([.5])
#     bounds = np.array([[.3, 1]])
#     eps = 1e-3
#     g = GaussianFieldRegressor(
#         Test_Weight(sigma, bounds),
#         optimizer=True,
#         smoothing=0,
#     )
#     g.set_model(X_train, y_train)
#     _, grad = g.cross_entropy(X_test, y_test, eval_gradient=True)

#     g.weight.theta += eps
#     f1 = g.cross_entropy(X_test, y_test)
#     g.weight.theta -= 2 * eps
#     f2 = g.cross_entropy(X_test, y_test)
#     g.weight.theta += eps
#     assert np.isclose((f1 - f2)/(2 * eps), grad[0])


# def test_laplacian_gradient():

#     class Test_Weight(Weight):
#         def __init__(self, sigma, bounds):
#             self.sigma = sigma
#             self._bounds = bounds

#         def __call__(self, X, Y=None, eval_gradient=False):
#             if Y is None:
#                 Y = X
#             m = pairwise_distances(X, Y)
#             w = np.exp(-(m/self.sigma[0])**2/2)
#             if eval_gradient:
#                 return w, np.array([m**2/self.sigma[0]**3]) * w
#             else:
#                 return w

#         @property
#         def theta(self):
#             return self.sigma

#         @theta.setter
#         def theta(self, values):
#             self.sigma = values

#         @property
#         def bounds(self):
#             return self._bounds

#     np.random.seed(0)
#     X_train = np.random.rand(20, 2)
#     y_train = np.random.rand(20)

#     sigma = np.array([.5])
#     bounds = np.array([[.3, 1]])
#     eps = 1e-3
#     g = GaussianFieldRegressor(
#         Test_Weight(sigma, bounds),
#         optimizer=True,
#         smoothing=0,
#     )
#     g.set_model(X_train, y_train)
#     _, grad = g.laplacian_error(eval_gradient=True)

#     g.weight.theta += eps
#     f1 = g.laplacian_error()
#     g.weight.theta -= 2 * eps
#     f2 = g.laplacian_error()
#     g.weight.theta += eps

#     assert np.isclose((f1 - f2)/(2 * eps), grad[0])


# def test_average_label_entropy_gradient():

#     class Test_Weight(Weight):
#         def __init__(self, sigma, bounds):
#             self.sigma = sigma
#             self._bounds = bounds

#         def __call__(self, X, Y=None, eval_gradient=False):
#             if Y is None:
#                 Y = X
#             m = pairwise_distances(X, Y)
#             w = np.exp(-(m/self.sigma[0])**2/2)
#             if eval_gradient:
#                 return w, np.array([m**2/self.sigma[0]**3]) * w
#             else:
#                 return w

#         @property
#         def theta(self):
#             return self.sigma

#         @theta.setter
#         def theta(self, values):
#             self.sigma = values

#         @property
#         def bounds(self):
#             return self._bounds

#     np.random.seed(0)
#     X_train = np.random.rand(20, 2)
#     y_train = np.random.rand(20)
#     X_test = np.random.rand(5, 2)
#     y_test = np.random.rand(5)

#     sigma = np.array([.5])
#     bounds = np.array([[.3, 1]])
#     eps = 1e-3
#     g = GaussianFieldRegressor(
#         Test_Weight(sigma, bounds),
#         optimizer=True,
#         smoothing=0,
#     )
#     g.set_model(X_train, y_train)
#     _, grad = g.average_label_entropy(X_test, y_test, eval_gradient=True)

#     g.weight.theta += eps
#     f1 = g.average_label_entropy(X_test, y_test)
#     g.weight.theta -= 2 * eps
#     f2 = g.average_label_entropy(X_test, y_test)
#     g.weight.theta += eps

#     assert np.isclose((f1 - f2)/(2 * eps), grad[0])


# def test_eta():
#     mock = MagicMock()
#     W = np.array([
#             [0, 1, 0.1],
#             [1, 0, 0.2],
#             [0.1, 0.2, 0],
#         ])
#     f_l = [1, 2]
#     mock.side_effect = [W]
#     g = GaussianFieldRegressor(mock, smoothing=0, eta=0.5)
#     g.set_model(np.zeros((2, 2)), f_l)
#     x = g.predict(np.array([[0, 0]]))
#     assert np.allclose(x, np.array([1.3392226, 1.6890459, 1.5724382]))


# def test_fit():

#     class Test_Weight(Weight):
#         def __init__(self, sigma, bounds):
#             self.sigma = sigma
#             self._bounds = bounds

#         def __call__(self, X, Y=None, eval_gradient=False):
#             if Y is None:
#                 Y = X
#             m = pairwise_distances(X, Y)
#             w = np.exp(-(m/self.sigma[0])**2/2)
#             if eval_gradient:
#                 return w, np.array([m**2/self.sigma[0]**3]) * w
#             else:
#                 return w

#         @property
#         def theta(self):
#             return self.sigma

#         @theta.setter
#         def theta(self, values):
#             self.sigma = values

#         @property
#         def bounds(self):
#             return self._bounds

#     np.random.seed(0)
#     X_train = np.random.rand(20, 2)
#     y_train = np.random.rand(20)
#     X_test = np.random.rand(5, 2)
#     y_test = np.random.rand(5)

#     sigma = np.array([1])

#     bounds = np.array([[1, 10]])
#     g = GaussianFieldRegressor(
#         Test_Weight(sigma, bounds),
#         optimizer=True,
#         smoothing=0,
#     )
#     g.set_model(X_train, y_train)
#     g.fit(Z=X_test, y=y_test)
#     assert g.weight.theta == np.array([10])
