#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pytest
from scipy.spatial import distance_matrix as pairwise_distances
from unittest.mock import MagicMock
from graphdot.model.gaussian_field import GaussianFieldRegressor
from graphdot.model.gaussian_field import Weight


def test_gaussian_field_fit_and_predict_before_set():
    """
    Tests that set_model must be called before fitting.
    """
    g = GaussianFieldRegressor(pairwise_distances)
    X = np.array([1])
    y = np.array([1])
    with pytest.raises(Exception):
        g.fit(X, y)
    with pytest.raises(Exception):
        g.predict(X, y)


def test_gaussian_field():
    """
    Tests Gaussian Field Regressor Prediction
    """
    class Test_Weight():

        def __call__(self, X, Y=None):
            if Y is None:
                return pairwise_distances(X, X)
            else:
                return pairwise_distances(X, Y)

    X = np.array([
            [1, 0],
            [0, 1],
            [-1, 0],
            [0, -1],
        ])
    y = np.array([1, 1, 1, 0])

    g = GaussianFieldRegressor(Test_Weight(), smoothing=0)
    g.set_model(X, y)

    Xu = np.array([[0, 0]])
    assert g.predict(Xu) == np.array([.75])


def test_gaussian_field_prediction1():
    """
    Tests Gaussian Field Regressor Prediction
    """
    mock = MagicMock()
    X = np.array([0, 1])
    y = np.array([0, 1])
    W_ul = np.array([
            [1, 0],
            [0, 0],
            [0, 1],
        ])
    W_uu = np.array([
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0],
        ])
    Xt = np.array([0, 1, 2])
    mock.side_effect = [W_ul, W_uu]
    g = GaussianFieldRegressor(mock, smoothing=0)
    g.set_model(X, y)
    A = g.predict(Xt)
    B = np.array([.25, .5, .75])
    assert np.allclose(A, B)


def test_gaussian_field_prediction2():
    """
    Tests Gaussian Field Regressor Prediction
    """
    mock = MagicMock()
    X = np.array([0])
    y = np.array([1])
    W_ul = np.array([
            [1],
            [1],
            [1],
            [1],
        ])
    W_uu = np.array([
            [0, 1, 1, 1],
            [1, 0, 1, 1],
            [1, 1, 0, 1],
            [1, 1, 1, 0],
        ])
    Xt = np.array([1, 2, 3, 4])
    mock.side_effect = [W_ul, W_uu]
    g = GaussianFieldRegressor(mock, smoothing=0)
    g.set_model(X, y)
    A = g.predict(Xt)
    B = np.array([1, 1, 1, 1])
    assert np.allclose(A, B)


def test_smoothing():
    """
    Tests Gaussian Field Regressor Prediction
    """
    mock = MagicMock()
    X = np.array([0, 1])
    y = np.array([0, 1])
    W_ul = np.array([
            [1, 0],
            [0, 0],
            [0, 1],
        ])
    W_uu = np.array([
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0],
        ])
    Xt = np.array([0, 1, 2])
    mock.side_effect = [W_ul, W_uu]
    g = GaussianFieldRegressor(mock, smoothing=1e-2)
    g.set_model(X, y)
    A = g.predict(Xt)
    B = np.array([.2525, .5, .7475])
    assert np.allclose(A, B)


def test_gaussian_field_influence_matrix():
    """
    Tests Gaussian Field Regressor Prediction
    """
    mock = MagicMock()
    X = np.array([0])
    y = np.array([1])
    W_ul = np.array([
            [1],
            [1],
            [1],
            [1],
        ])
    W_uu = np.array([
            [0, 1, 1, 1],
            [1, 0, 1, 1],
            [1, 1, 0, 1],
            [1, 1, 1, 0],
        ])
    Xt = np.array([1, 2, 3, 4])
    mock.side_effect = [W_ul, W_uu]
    g = GaussianFieldRegressor(mock, smoothing=0)
    g.set_model(X, y)
    _, A, _ = g.predict(Xt, display=True)
    B = np.array([1, 1, 1, 1])
    assert np.allclose(np.dot(A, y), B)


def test_squared_error_gradient():

    class Test_Weight(Weight):
        def __init__(self, sigma, bounds):
            self.sigma = sigma
            self._bounds = bounds

        def __call__(self, X, Y=None, eval_gradient=False):
            if Y is None:
                Y = X
            m = pairwise_distances(X, Y)
            w = np.exp(-(m/self.sigma[0])**2/2)
            if eval_gradient:
                return w, np.array([m**2/self.sigma[0]**3]) * w
            else:
                return w

        @property
        def theta(self):
            return self.sigma

        @theta.setter
        def theta(self, values):
            self.sigma = values

        @property
        def bounds(self):
            return self._bounds

    np.random.seed(0)
    X_train = np.random.rand(20, 2)
    y_train = np.random.rand(20)
    X_test = np.random.rand(5, 2)
    y_test = np.random.rand(5)

    sigma = np.array([.5])
    bounds = np.array([[.3, 1]])
    eps = 1e-3
    g = GaussianFieldRegressor(
        Test_Weight(sigma, bounds),
        optimizer=True,
        smoothing=0,
    )
    g.set_model(X_train, y_train)
    _, grad = g.squared_error(X_test, y_test, eval_gradient=True)

    g.weight.theta += eps
    f1 = g.squared_error(X_test, y_test)
    g.weight.theta -= 2 * eps
    f2 = g.squared_error(X_test, y_test)
    g.weight.theta += eps
    assert np.isclose((f1 - f2)/(2 * eps), grad[0])


def test_cross_entropy_gradient():

    class Test_Weight(Weight):
        def __init__(self, sigma, bounds):
            self.sigma = sigma
            self._bounds = bounds

        def __call__(self, X, Y=None, eval_gradient=False):
            if Y is None:
                Y = X
            m = pairwise_distances(X, Y)
            w = np.exp(-(m/self.sigma[0])**2/2)
            if eval_gradient:
                return w, np.array([m**2/self.sigma[0]**3]) * w
            else:
                return w

        @property
        def theta(self):
            return self.sigma

        @theta.setter
        def theta(self, values):
            self.sigma = values

        @property
        def bounds(self):
            return self._bounds

    np.random.seed(0)
    X_train = np.random.rand(20, 2)
    y_train = np.random.rand(20)
    X_test = np.random.rand(5, 2)
    y_test = np.random.rand(5)

    sigma = np.array([.5])
    bounds = np.array([[.3, 1]])
    eps = 1e-3
    g = GaussianFieldRegressor(
        Test_Weight(sigma, bounds),
        optimizer=True,
        smoothing=0,
    )
    g.set_model(X_train, y_train)
    _, grad = g.cross_entropy(X_test, y_test, eval_gradient=True)

    g.weight.theta += eps
    f1 = g.cross_entropy(X_test, y_test)
    g.weight.theta -= 2 * eps
    f2 = g.cross_entropy(X_test, y_test)
    g.weight.theta += eps
    assert np.isclose((f1 - f2)/(2 * eps), grad[0])


def test_laplacian_gradient():

    class Test_Weight(Weight):
        def __init__(self, sigma, bounds):
            self.sigma = sigma
            self._bounds = bounds

        def __call__(self, X, Y=None, eval_gradient=False):
            if Y is None:
                Y = X
            m = pairwise_distances(X, Y)
            w = np.exp(-(m/self.sigma[0])**2/2)
            if eval_gradient:
                return w, np.array([m**2/self.sigma[0]**3]) * w
            else:
                return w

        @property
        def theta(self):
            return self.sigma

        @theta.setter
        def theta(self, values):
            self.sigma = values

        @property
        def bounds(self):
            return self._bounds

    np.random.seed(0)
    X_train = np.random.rand(20, 2)
    y_train = np.random.rand(20)

    sigma = np.array([.5])
    bounds = np.array([[.3, 1]])
    eps = 1e-3
    g = GaussianFieldRegressor(
        Test_Weight(sigma, bounds),
        optimizer=True,
        smoothing=0,
    )
    g.set_model(X_train, y_train)
    _, grad = g.laplacian_error(eval_gradient=True)

    g.weight.theta += eps
    f1 = g.laplacian_error()
    g.weight.theta -= 2 * eps
    f2 = g.laplacian_error()
    g.weight.theta += eps

    assert np.isclose((f1 - f2)/(2 * eps), grad[0])


def test_average_label_entropy_gradient():

    class Test_Weight(Weight):
        def __init__(self, sigma, bounds):
            self.sigma = sigma
            self._bounds = bounds

        def __call__(self, X, Y=None, eval_gradient=False):
            if Y is None:
                Y = X
            m = pairwise_distances(X, Y)
            w = np.exp(-(m/self.sigma[0])**2/2)
            if eval_gradient:
                return w, np.array([m**2/self.sigma[0]**3]) * w
            else:
                return w

        @property
        def theta(self):
            return self.sigma

        @theta.setter
        def theta(self, values):
            self.sigma = values

        @property
        def bounds(self):
            return self._bounds

    np.random.seed(0)
    X_train = np.random.rand(20, 2)
    y_train = np.random.rand(20)
    X_test = np.random.rand(5, 2)
    y_test = np.random.rand(5)

    sigma = np.array([.5])
    bounds = np.array([[.3, 1]])
    eps = 1e-3
    g = GaussianFieldRegressor(
        Test_Weight(sigma, bounds),
        optimizer=True,
        smoothing=0,
    )
    g.set_model(X_train, y_train)
    _, grad = g.average_label_entropy(X_test, y_test, eval_gradient=True)

    g.weight.theta += eps
    f1 = g.average_label_entropy(X_test, y_test)
    g.weight.theta -= 2 * eps
    f2 = g.average_label_entropy(X_test, y_test)
    g.weight.theta += eps

    assert np.isclose((f1 - f2)/(2 * eps), grad[0])


def test_eta():
    mock = MagicMock()
    W = np.array([
            [0, 1, 0.1],
            [1, 0, 0.2],
            [0.1, 0.2, 0],
        ])
    f_l = [1, 2]
    mock.side_effect = [W]
    g = GaussianFieldRegressor(mock, smoothing=0, eta=0.5)
    g.set_model(np.zeros((2, 2)), f_l)
    x = g.predict(np.array([[0, 0]]))
    assert np.allclose(x, np.array([1.3392226, 1.6890459, 1.5724382]))


def test_fit():

    class Test_Weight(Weight):
        def __init__(self, sigma, bounds):
            self.sigma = sigma
            self._bounds = bounds

        def __call__(self, X, Y=None, eval_gradient=False):
            if Y is None:
                Y = X
            m = pairwise_distances(X, Y)
            w = np.exp(-(m/self.sigma[0])**2/2)
            if eval_gradient:
                return w, np.array([m**2/self.sigma[0]**3]) * w
            else:
                return w

        @property
        def theta(self):
            return self.sigma

        @theta.setter
        def theta(self, values):
            self.sigma = values

        @property
        def bounds(self):
            return self._bounds

    np.random.seed(0)
    X_train = np.random.rand(20, 2)
    y_train = np.random.rand(20)
    X_test = np.random.rand(5, 2)
    y_test = np.random.rand(5)

    sigma = np.array([1])

    bounds = np.array([[1, 10]])
    g = GaussianFieldRegressor(
        Test_Weight(sigma, bounds),
        optimizer=True,
        smoothing=0,
    )
    g.set_model(X_train, y_train)
    g.fit(Z=X_test, y=y_test)
    assert g.weight.theta == np.array([10])
