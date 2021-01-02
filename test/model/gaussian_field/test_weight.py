#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest
import numpy as np
from scipy.spatial.distance import pdist, squareform
from graphdot.model.gaussian_field import RBFOverDistance, RBFOverFixedDistance


class CosineMetric:
    def __init__(self, xi):
        self.xi = xi

    def __call__(self, X, Y=None, eval_gradient=False):
        Y = X if Y is None else Y
        K1 = np.sum(X**2, axis=1)
        K2 = np.sum(Y**2, axis=1)
        K12 = X @ Y.T
        K = np.diag(K1**-0.5) @ K12 @ np.diag(K2**-0.5)
        D = np.sqrt(np.maximum(1e-15, 2 - 2 * K**self.xi))
        if eval_gradient is True:
            grad = -np.log(K) * K**self.xi / D
            return D, grad.reshape(*grad.shape, 1)
        else:
            return D

    @property
    def theta(self):
        return np.log([self.xi])

    @theta.setter
    def theta(self, values):
        self.xi = np.exp(values[0])

    @property
    def bounds(self):
        return np.log([[0.1, 10.0]])


def test_rbf_over_fixed_distance():

    sigma = 1.0
    n = 16
    k = 3
    weight = RBFOverFixedDistance(
        D=squareform(pdist(np.random.randn(n, k))),
        sigma=sigma,
        sigma_bounds=(0.1, 2.0)
    )

    assert isinstance(weight.theta, np.ndarray)
    assert weight.theta.ndim == 1
    assert weight.theta[0] == np.log(sigma)
    weight.theta = [0.1]
    assert weight.theta[0] == pytest.approx(0.1)

    assert isinstance(weight.bounds, np.ndarray)
    assert weight.bounds.ndim == 2
    assert weight.bounds.shape == (1, 2)

    for i in range(2, n):
        X = np.arange(i)
        Y = np.arange(i, n)
        assert np.allclose(weight(X).diagonal(), 1)
        assert np.all(weight(X, Y) > 0)
        assert np.all(weight(X, Y) < 1)


@pytest.mark.parametrize('sigma', [0.5, 1.0, 2.0])
def test_rbf_over_fixed_distance_gradient(sigma):
    eps = 1e-4
    n = 16
    k = 3
    weight = RBFOverFixedDistance(
        D=squareform(pdist(np.random.randn(n, k))),
        sigma=sigma,
    )

    X = np.arange(n)
    _, dW = weight(X, eval_gradient=True)
    for i, _ in enumerate(weight.theta):
        pos, neg = np.exp(weight.theta), np.exp(weight.theta)
        pos[i] += eps
        neg[i] -= eps
        w_pos = weight.clone_with_theta(np.log(pos))(X)
        w_neg = weight.clone_with_theta(np.log(neg))(X)
        assert np.allclose(
            (w_pos - w_neg) / (2 * eps),
            dW[:, :, i],
            rtol=1e-5,
            atol=1e-5
        )


def test_rbf_over_distance():

    xi = 2.0
    sigma = 1.0
    metric = CosineMetric(xi)
    weight = RBFOverDistance(metric=metric, sigma=sigma)

    assert isinstance(weight.theta, np.ndarray)
    assert weight.theta.ndim == 1
    assert len(weight.theta) == 1 + len(metric.theta)
    assert isinstance(weight.bounds, np.ndarray)
    assert weight.bounds.ndim == 2
    assert len(weight.bounds) == len(metric.bounds) + 1

    for n in range(3, 9):
        for k in range(2, 7):
            X = np.random.rand(n, k)
            Y = np.random.rand(n // 2, k)
            assert np.allclose(weight(X).diagonal(), 1)
            assert np.allclose(weight(Y).diagonal(), 1)
            assert np.all(weight(X, Y) > 0)
            assert np.all(weight(X, Y) < 1)


@pytest.mark.parametrize('xi', [1.0, 2.0, 3.0])
@pytest.mark.parametrize('sigma', [0.5, 1.0, 2.0])
def test_rbf_over_distance_gradient(xi, sigma):

    metric = CosineMetric(xi)
    weight = RBFOverDistance(metric=metric, sigma=sigma)

    X = np.random.rand(8, 2)

    np.set_printoptions(precision=5, linewidth=270, suppress=True)
    _, dW = weight(X, eval_gradient=True)
    eps = 1e-4
    for i, _ in enumerate(weight.theta):
        pos, neg = np.exp(weight.theta), np.exp(weight.theta)
        pos[i] += eps
        neg[i] -= eps
        w_pos = weight.clone_with_theta(np.log(pos))(X)
        w_neg = weight.clone_with_theta(np.log(neg))(X)
        diff = (w_pos - w_neg) / (2 * eps)
        assert np.allclose(
            diff,
            dW[:, :, i],
            rtol=1e-6,
            atol=1e-6
        )
