#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest
import numpy as np
from graphdot.linalg.block import binvh1


@pytest.mark.parametrize('seed', [0, 51, 125, 125120, 9681859])
@pytest.mark.parametrize('sigma', np.logspace(-2, 1, 7))
@pytest.mark.parametrize('size', [1, 2, 3, 4, 5, 7, 13, 16, 22, 31, 64, 215])
def test_binvh1(seed, sigma, size):

    def kernel(X, Y=None, sigma=1.0):
        return np.exp(-0.5 * np.subtract.outer(X, Y or X)**2 / sigma**2)

    np.random.seed(seed)

    X = np.sort(np.random.rand(size))
    x = np.random.rand(1)
    Y = np.concatenate((X, x))
    alpha = 1e-3

    Kinv = binvh1(
        np.linalg.inv(kernel(X, sigma=sigma) + alpha * np.eye(len(X))),
        kernel(X, x, sigma=sigma).ravel(),
        kernel(x, x, sigma=sigma).item() + alpha
    )

    eye = Kinv @ (kernel(Y, sigma=sigma) + alpha * np.eye(len(Y)))

    assert(np.abs(eye - np.eye(len(Y))).max() < 1e-7)
