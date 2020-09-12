#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest
import numpy as np
from graphdot.model.active_learning import VarianceMinimizer


@pytest.mark.parametrize('N', [3, 5, 7, 9, 11, 13, 17, 33, 47, 85, 99])
def test_pick_line(N):

    class Kernel:
        def __init__(self, s):
            self.s = s

        def __call__(self, X, Y=None):
            return np.exp(
                -np.subtract.outer(X, Y if Y is not None else X)**2 / self.s**2
            )

    kernel = Kernel(2.0)
    X = np.linspace(-2, 2, N)
    selector = VarianceMinimizer(kernel)

    assert(selector(X, 1)[0] == N // 2)
    assert(np.sort(selector(X, 3)).tolist() == [0, N//2, N - 1])


@pytest.mark.parametrize('N', [4, 6, 8, 12, 22, 32, 48, 86])
def test_pick_ring(N):

    class Kernel:
        def __call__(self, X, Y=None):
            d = np.subtract.outer(X, Y if Y is not None else X)
            return np.exp(-2 * np.sin(np.pi / 0.5 * d)**2)

        def diag(self, X):
            return np.ones_like(X)

    kernel = Kernel()
    X = np.linspace(0, 0.5, N, endpoint=False)
    selector = VarianceMinimizer(kernel)

    chosen = selector(X, 2)
    diff = (chosen[0] - chosen[1] + N) % N
    assert(abs(diff) == N // 2)


@pytest.mark.parametrize('N', [2, 5, 7, 10, 61, 285, 1985])
def test_pick_count(N):

    class Kernel:
        def __init__(self, s):
            self.s = s

        def __call__(self, X, Y=None):
            return np.exp(
                -np.subtract.outer(X, Y if Y is not None else X)**2 / self.s**2
            )

    kernel = Kernel(3.0)
    X = np.random.rand(N) * N
    selector = VarianceMinimizer(kernel)

    for n in np.random.choice(min(N, 100), min(N, 5), False):
        chosen = selector(X, n)
        assert(len(np.unique(chosen)) == n)


def test_pick_no_duplicates():

    class Kernel:
        def __init__(self, s):
            self.s = s

        def __call__(self, X, Y=None):
            return np.exp(
                -np.subtract.outer(X, Y if Y is not None else X)**2 / self.s**2
            )

    kernel = Kernel(1.0)
    N = 3
    X = np.zeros(N)
    selector = VarianceMinimizer(kernel)

    chosen = selector(X, N)
    assert(len(np.unique(chosen)) == 3)


@pytest.mark.parametrize('N', [96, 241, 1523])
def test_pick_superiority(N):

    class Kernel:
        def __init__(self, s):
            self.s = s

        def __call__(self, X, Y=None):
            return np.exp(
                -np.subtract.outer(X, Y if Y is not None else X)**2 / self.s**2
            )

    kernel = Kernel(3.0)
    upper = 100
    X = np.random.rand(N) * upper
    selector = VarianceMinimizer(kernel)

    for n in [10, 20, 50]:
        chosen = selector(X, n)
        random = np.random.choice(len(X), n, False)
        det_active = np.prod(np.linalg.slogdet(kernel(X[chosen])))
        det_random = np.prod(np.linalg.slogdet(kernel(X[random])))
        assert(det_active > det_random)
