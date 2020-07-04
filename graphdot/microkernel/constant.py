#!/usr/bin/env python
# -*- coding: utf-8 -*-
from collections import namedtuple
import numpy as np
from graphdot.codegen.cpptool import cpptype
from ._base import MicroKernel


def Constant(c, c_bounds=None):
    r"""Creates a no-op microkernel that returns a constant value,
    i.e. :math:`k_\mathrm{c}(\cdot, \cdot) \equiv constant`. This kernel is
    often mutliplied with other microkernels as an adjustable weight.

    Parameters
    ----------
    c: float > 0
        The constant value.
    """
    if c_bounds is None:
        c_bounds = (c, c)

    @cpptype(c=np.float32)
    class ConstantKernel(MicroKernel):
        @property
        def name(self):
            return 'Constant'

        def __init__(self, c, c_bounds):
            self.c = float(c)
            self.c_bounds = c_bounds
            self._assert_bounds('c', c_bounds)

        def __call__(self, i, j, jac=False):
            if jac is True:
                return self.c, np.ones(1)
            else:
                return self.c

        def __repr__(self):
            return f'{self.name}({self.c})'

        def gen_expr(self, x, y, jac=False, theta_scope=''):
            f = f'{theta_scope}c'
            if jac is True:
                return f, ['1.0f']
            else:
                return f

        @property
        def theta(self):
            return namedtuple(
                f'{self.name}Hyperparameters',
                ['c']
            )(self.c)

        @theta.setter
        def theta(self, seq):
            self.c = seq[0]

        @property
        def bounds(self):
            return (self.c_bounds,)

    return ConstantKernel(c, c_bounds)
