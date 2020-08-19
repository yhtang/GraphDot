#!/usr/bin/env python
# -*- coding: utf-8 -*-
from collections import namedtuple
import numpy as np
from graphdot.codegen.cpptool import cpptype
from ._base import MicroKernel


def KroneckerDelta(h, h_bounds=(1e-3, 1)):
    r"""Creates a Kronecker delta microkernel that returns either 1 or h
    depending on whether two features compare equal, i.e. :math:`k_\delta(i, j)
    = \begin{cases} 1, i = j \\ h, otherwise \end{cases}`.

    Parameters
    ----------
    h: float in (0, 1)
        The value of the microkernel when two features do not compare equal.
    h_bounds: tuple or "fixed"
        If tuple, contains the lower and upper bounds that `h` is allowed to
        vary during hyperparameter optimization. If "fixed", the hyperparameter
        will not be optimized during training.
    """

    @cpptype(h=np.float32)
    class KroneckerDeltaKernel(MicroKernel):

        @property
        def name(self):
            return 'KroneckerDelta'

        def __init__(self, h, h_bounds):
            self.h = float(h)
            self.h_bounds = h_bounds
            self._assert_bounds('h', h_bounds)

        def __call__(self, i, j, jac=False):
            if jac is True:
                return (
                    1.0 if i == j else self.h,
                    np.array([0.0 if i == j else 1.0])
                )
            else:
                return 1.0 if i == j else self.h

        def __repr__(self):
            return f'{self.name}({self.h})'

        def gen_expr(self, x, y, theta_scope=''):
            f = f'({x} == {y} ? 1.0f : {theta_scope}h)'
            j = [f'({x} == {y} ? 0.0f : 1.0f)']
            return f, j

        @property
        def theta(self):
            return namedtuple(
                f'{self.name}Hyperparameters',
                ['h']
            )(self.h)

        @theta.setter
        def theta(self, seq):
            self.h = seq[0]

        @property
        def bounds(self):
            return (self.h_bounds,)

    return KroneckerDeltaKernel(h, h_bounds)
