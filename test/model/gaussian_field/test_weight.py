#!/usr/bin/env python
# -*- coding: utf-8 -*-
from unittest.mock import Mock
import pytest
import numpy as np
from scipy.spatial import distance_matrix as metric
from graphdot.model.gaussian_field import RBFOverDistance


def test_rbf_over_distance():

    weight = RBFOverDistance(
        metric=lambda X, Y=None:
            metric(X, X) if Y is None else metric(X, Y),
        sigma=1.0,
        sigma_bounds=(0.1, 2.0)
    )

    assert isinstance(weight.theta, np.ndarray)
    assert weight.theta.ndim == 1
    assert weight.theta[0] == np.log(1.0)
    weight.theta = [0.1]
    assert weight.theta[0] == pytest.approx(0.1)

    assert isinstance(weight.bounds, np.ndarray)
    assert weight.bounds.ndim == 2
    assert weight.bounds.shape == (1, 2)

    for d in range(2, 9):
        X = np.zeros((1, d))
        Y = np.ones((1, d))
        weight.theta = np.log([np.sqrt(d)])
        assert np.allclose(weight(X), np.ones((1, 1)))
        assert np.allclose(weight(X, Y), np.ones((1, 1)) * np.exp(-0.5))

    '''test gradient'''
    eps = 1e-4
    X = np.random.randn(4, 3)
    for i, _ in enumerate(weight.theta):
        weight.theta = [0]
        w, dw = weight(X, eval_gradient=True)
        theta_pos = np.copy(weight.theta)
        theta_neg = np.copy(weight.theta)
        theta_pos[i] += eps
        theta_neg[i] -= eps
        weight.theta = theta_pos
        w_pos = weight(X)
        weight.theta = theta_neg
        w_neg = weight(X)
        assert np.allclose(
            (w_pos - w_neg) / (2 * eps),
            dw,
            rtol=1e-5,
            atol=1e-5
        )


def test_rbf_over_distance_sticky_cache():

    m = Mock(
        side_effect=lambda X, Y=None:
            metric(X, X) if Y is None else metric(X, Y)
    )

    weight = RBFOverDistance(
        metric=m,
        sigma=1.0,
        sigma_bounds=(0.1, 2.0),
        sticky_cache=True
    )

    X = np.random.randn(4, 3)
    Y = np.random.randn(6, 3)

    weight(X)
    m.assert_called_once()
    m.reset_mock()
    weight(X)
    m.assert_not_called()
    m.reset_mock()
    weight(Y)
    m.assert_not_called()
    m.reset_mock()

    weight(X, Y)
    m.assert_called_once()
    m.reset_mock()
    weight(X, Y)
    m.assert_not_called()
    m.reset_mock()
    weight(Y, X)
    m.assert_not_called()
    m.reset_mock()
