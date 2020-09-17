#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from graphdot.codegen.cpptool import cpptype
from ._base import MicroKernel


def DotProduct():
    r"""Creates a dot product microkernel, which computes the inner product
    between two vector-valued features.

    Parameters
    ----------
    This kernel does not have any tunable hyperparameters.
    """

    @cpptype([])
    class DotProductKernel(MicroKernel):
        @property
        def name(self):
            return 'DotProduct'

        def __call__(self, X, Y, jac=False):
            X = np.asarray(X)
            Y = np.asarray(Y)
            if jac is True:
                return X @ Y, []
            else:
                return X @ Y

        def __repr__(self):
            return f'{self.name}()'

        def gen_expr(self, x, y, theta_scope=''):
            return f'dotproduct({x}, {y})', []

        @property
        def theta(self):
            return tuple()

        @theta.setter
        def theta(self, seq):
            pass

        @property
        def bounds(self):
            return tuple()

        @property
        def minmax(self):
            return (0, np.inf)

    return DotProductKernel()
