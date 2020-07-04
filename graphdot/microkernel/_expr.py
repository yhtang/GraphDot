#!/usr/bin/env python
# -*- coding: utf-8 -*-
from abc import abstractmethod
import numpy as np
from graphdot.codegen.cpptool import cpptype
from ._base import MicroKernel
from .constant import Constant


class MicroKernelExpr(MicroKernel):

    @property
    @abstractmethod
    def opstr(self):
        pass

    def __init__(self, k1, k2):
        self.k1 = k1
        self.k2 = k2

    def __repr__(self):
        return f'{repr(self.k1)} {self.opstr} {repr(self.k2)}'

    @property
    def theta(self):
        return (self.k1.theta, self.k2.theta)

    @theta.setter
    def theta(self, seq):
        self.k1.theta = seq[0]
        self.k2.theta = seq[1]

    @property
    def bounds(self):
        return (self.k1.bounds, self.k2.bounds)

    @staticmethod
    def add(k1, k2):
        k1 = k1 if isinstance(k1, MicroKernel) else Constant(k1)
        k2 = k2 if isinstance(k2, MicroKernel) else Constant(k2)

        @cpptype(k1=k1.dtype, k2=k2.dtype)
        class Add(MicroKernelExpr):

            @property
            def opstr(self):
                return '+'

            @property
            def name(self):
                return 'Add'

            def __call__(self, i, j, jac=False):
                if jac is True:
                    f1, J1 = self.k1(i, j, True)
                    f2, J2 = self.k2(i, j, True)
                    return (f1 + f2, J1 + J2)
                else:
                    return self.k1(i, j, False) + self.k2(i, j, False)

            def gen_expr(self, x, y, jac=False, theta_scope=''):
                if jac is True:
                    f1, J1 = self.k1.gen_expr(x, y, True, theta_scope + 'k1.')
                    f2, J2 = self.k2.gen_expr(x, y, True, theta_scope + 'k2.')
                    return (f'({f1} + {f2})', J1 + J2)
                else:
                    f1 = self.k1.gen_expr(x, y, False, theta_scope + 'k1.')
                    f2 = self.k2.gen_expr(x, y, False, theta_scope + 'k2.')
                    return f'({f1} + {f2})'

        return Add(k1, k2)

    @staticmethod
    def mul(k1, k2):
        k1 = k1 if isinstance(k1, MicroKernel) else Constant(k1)
        k2 = k2 if isinstance(k2, MicroKernel) else Constant(k2)

        @cpptype(k1=k1.dtype, k2=k2.dtype)
        class Multiply(MicroKernelExpr):

            @property
            def opstr(self):
                return '*'

            @property
            def name(self):
                return 'Multiply'

            def __call__(self, i, j, jac=False):
                if jac is True:
                    f1, J1 = self.k1(i, j, True)
                    f2, J2 = self.k2(i, j, True)
                    return (
                        f1 * f2,
                        np.array(
                            [j1 * f2 for j1 in J1] + [f1 * j2 for j2 in J2]
                        )
                    )
                else:
                    return self.k1(i, j, False) * self.k2(i, j, False)

            def gen_expr(self, x, y, jac=False, theta_scope=''):
                if jac is True:
                    f1, J1 = self.k1.gen_expr(x, y, True, theta_scope + 'k1.')
                    f2, J2 = self.k2.gen_expr(x, y, True, theta_scope + 'k2.')
                    return (
                        f'({f1} * {f2})',
                        [f'({j1} * {f2})' for j1 in J1] +
                        [f'({f1} * {j2})' for j2 in J2]
                    )
                else:
                    f1 = self.k1.gen_expr(x, y, False, theta_scope + 'k1.')
                    f2 = self.k2.gen_expr(x, y, False, theta_scope + 'k2.')
                    return f'({f1} * {f2})'

        return Multiply(k1, k2)

    @staticmethod
    def pow(k1, c):
        if isinstance(c, (int, float)):
            k2 = Constant(c)
        elif isinstance(c, MicroKernel) and c.name == 'Constant':
            k2 = c
        else:
            raise ValueError(
                f'Exponent must be a constant or constant microkernel, '
                f'got {c} instead.'
            )

        @cpptype(k1=k1.dtype, k2=k2.dtype)
        class Exponentiation(MicroKernelExpr):

            @property
            def opstr(self):
                return '**'

            @property
            def name(self):
                return 'Exponentiation'

            def __call__(self, i, j, jac=False):
                if jac is True:
                    # d(x^y) / dx = y * x^(y - 1)
                    # d(x^y) / dy = x^y * log(x)
                    f1, J1 = self.k1(i, j, True)
                    f2, J2 = self.k2(i, j, True)
                    return (
                        f1**f2,
                        np.array(
                            [f2 * f1**(f2 - 1) * j1 for j1 in J1] +
                            [f1**f2 * np.log(f1) * j2 for j2 in J2]
                        )
                    )
                else:
                    return self.k1(i, j, False)**self.k2(i, j, False)

            def gen_expr(self, x, y, jac=False, theta_scope=''):
                if jac is True:
                    f1, J1 = self.k1.gen_expr(x, y, True, theta_scope + 'k1.')
                    f2, J2 = self.k2.gen_expr(x, y, True, theta_scope + 'k2.')
                    return (
                        f'__powf({f1}, {f2})',
                        [f'({f2} * __powf({f1}, {f2} - 1) * {j})'
                         for j in J1] +
                        [f'(__powf({f1}, {f2}) * __logf({f1}) * {j})'
                         for j in J2]
                    )
                else:
                    f1 = self.k1.gen_expr(x, y, False, theta_scope + 'k1.')
                    f2 = self.k2.gen_expr(x, y, False, theta_scope + 'k2.')
                    return f'__powf({f1}, {f2})'

        return Exponentiation(k1, k2)
