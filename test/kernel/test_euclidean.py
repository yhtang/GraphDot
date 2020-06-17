#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest
import numpy as np
from graphdot.kernel.euclidean import RBFKernel


def test_simple():

    kernel = RBFKernel(
        's**2 * exp(-2 * d**2 / L**2)',
        'd',
        s=1.0,
        L=1.0,
    )

    assert(kernel.get_params()['s'] == 1.0)
    assert(kernel.get_params()['L'] == 1.0)
    assert(kernel.theta == pytest.approx(np.zeros(2)))
    kernel.theta = np.zeros(2)

    for x in np.random.randn(1000):
        assert(kernel([[x]]) == np.ones((1, 1)))
        assert(kernel([[x]], [[x]]) == np.ones((1, 1)))
    for x, y in np.random.randn(1000, 2):
        assert(kernel([[x, y]]) == np.ones((1, 1)))
        assert(kernel([[x, y]], [[x, y]]) == np.ones((1, 1)))

    for n in range(1, 10):
        assert(len(kernel.diag([[0] * n])) == 1)
        assert(kernel.diag([[0]] * n) == pytest.approx(np.ones(n)))

    assert(len(kernel.gradient([[0]])) == 2)
