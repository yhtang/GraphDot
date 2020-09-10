#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest
import numpy as np
from graphdot.linalg.spectral import pinvh


@pytest.mark.parametrize('seed', [0, 51, 125, 125120, 9681859])
@pytest.mark.parametrize('sigma', np.logspace(-2, 1, 11))
@pytest.mark.parametrize('mode', ['truncate', 'clamp'])
def test_pinvh(seed, sigma, mode):

    def kernel(X, sigma):
        return np.exp(-0.5 * np.subtract.outer(X, X)**2 / sigma**2)

    np.random.seed(seed)

    X = np.sort(np.random.rand(64))
    y = np.cos(4.0 * X * np.pi)

    alpha = 1e-10
    rcond = 1e-10
    eps = np.random.rand(len(X), len(X)) * 1e-7
    eps = eps + eps.T

    K = kernel(X, sigma=sigma)
    K += eps
    K.flat[::len(X) + 1] += alpha

    K_inv, logdetK = pinvh(K, rcond=rcond, mode=mode, return_nlogdet=True)

    assert(K @ K_inv @ K == pytest.approx(K, abs=5e-5))
    assert(logdetK < 0)
    assert(y @ (K_inv @ y) > 0)
