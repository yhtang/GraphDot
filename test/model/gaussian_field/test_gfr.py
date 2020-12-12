#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pytest
from scipy.spatial import distance_matrix as pairwise_distances
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

