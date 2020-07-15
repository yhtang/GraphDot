#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from graphdot.codegen.cpptool import cpptype
from ._base import MicroKernel


@cpptype([])
class Product(MicroKernel):
    """Direct product between features."""

    @property
    def name(self):
        return 'Product'

    def __call__(self, x1, x2, jac=False):
        if jac is True:
            return x1 * x2, np.array([])
        else:
            return x1 * x2

    def __repr__(self):
        return f'{self.name}()'

    def gen_expr(self, x, y, theta_scope=''):
        return f'({x} * {y})', []

    @property
    def theta(self):
        return tuple()

    @theta.setter
    def theta(self, seq):
        pass

    @property
    def bounds(self):
        return tuple()
