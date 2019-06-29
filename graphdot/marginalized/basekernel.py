#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module defines base kernels and composibility rules for creating vertex
and edges kernels for the marginalized graph kernel.
"""
from copy import copy
import numpy as np
from graphdot.codegen.typetool import cpptype

__all__ = ['Constant',
           'KroneckerDelta',
           'SquareExponential',
           'TensorProduct']

__cpp_namespace__ = 'graphdot::basekernel'


class Kernel(object):
    """
    Parent class for all base kernels
    """

    def __add__(self, b):
        return Kernel._op(self, b if isinstance(b, Kernel) else Constant(b),
                          lambda x, y: x + y, '+')

    def __radd__(self, b):
        return Kernel._op(b if isinstance(b, Kernel) else Constant(b), self,
                          lambda x, y: x + y, '+')

    def __mul__(self, b):
        return Kernel._op(self, b if isinstance(b, Kernel) else Constant(b),
                          lambda x, y: x * y, '*')

    def __rmul__(self, b):
        return Kernel._op(b if isinstance(b, Kernel) else Constant(b), self,
                          lambda x, y: x * y, '*')

    @staticmethod
    def _op(k1, k2, op, opstr):
        @cpptype(k1=k1.dtype, k2=k2.dtype)
        class KernelOperator:
            def __init__(self, k1, k2):
                self.k1 = copy(k1)
                self.k2 = copy(k2)

            def __call__(self, i, j):
                return op(self.k1(i, j), self.k2(i, j))

            def __repr__(self):
                return '({} {} {})'.format(repr(self.k1), opstr, repr(self.k2))

            def gencode(self, x, y):
                return '({k1} {op} {k2})'.format(k1=self.k1.gencode(x, y),
                                                 k2=self.k2.gencode(x, y),
                                                 op=opstr)

            @property
            def theta(self):
                return [self.k1.theta, self.k2.theta]

        return KernelOperator(k1, k2)


@cpptype(constant=np.float32)
class Constant(Kernel):
    def __init__(self, constant):
        self.constant = float(constant)

    def __call__(self, i, j):
        return self.constant

    def __repr__(self):
        return '{}'.format(self.constant)

    def gencode(self, x, y):
        return '{:f}f'.format(self.constant)

    @property
    def theta(self):
        return [self.constant]


@cpptype(lo=np.float32, hi=np.float32)
class KroneckerDelta(Kernel):

    def __init__(self, lo, hi=1.0):
        self.lo = float(lo)
        self.hi = float(hi)

    def __call__(self, i, j):
        return self.hi if i == j else self.lo

    def __repr__(self):
        return 'δ({}, {})'.format(self.hi, self.lo)

    def gencode(self, x, y):
        return '({} == {} ? {:f}f : {:f}f)'.format(x, y, self.hi, self.lo)

    @property
    def theta(self):
        return [self.lo, self.hi]


@cpptype(neg_half_inv_l2=np.float32)
class SquareExponential(Kernel):
    def __init__(self, length_scale):
        self.length_scale = length_scale

    def __call__(self, x1, x2):
        return np.exp(-0.5 * np.sum((x1 - x2)**2) / self.length_scale**2)

    def __repr__(self):
        return 'SqExp({})'.format(self.length_scale)

    def gencode(self, x, y):
        return 'expf({:f}f * power({} - {}, 2))'.format(
            -0.5 / self.length_scale**2, x, y)

    @property
    def theta(self):
        return [self.length_scale]

    @property
    def neg_half_inv_l2(self):
        return -0.5 / self.length_scale**2


def TensorProduct(**kw_kernels):
    @cpptype([(key, ker.dtype) for key, ker in kw_kernels.items()])
    class TensorProductKernel(Kernel):
        def __init__(self, **kw_kernels):
            self.kw_kernels = kw_kernels

        def __call__(self, object1, object2):
            prod = 1.0
            for key, kernel in self.kw_kernels.items():
                prod *= kernel(object1[key], object2[key])
            return prod

        def __repr__(self):
            return ' ⊗ '.join([kw + ':' + repr(k)
                               for kw, k in self.kw_kernels.items()])

        def gencode(self, x, y):
            return ' * '.join([k.gencode('%s.%s' % (x, key),
                                         '%s.%s' % (y, key))
                               for key, k in self.kw_kernels.items()])

        @property
        def theta(self):
            return [k.theta for k in self.kw_kernels.values()]

    return TensorProductKernel(**kw_kernels)

# class Convolution(Kernel):
#     def __init__(self, kernel):
#         self.kernel = kernel
#
#     def __call__(self, object1, object2):
#         sum = 0.0
#         for part1 in object1:
#             for part2 in object2:
#                 sum += self.kernel(part1, part2)
#         return sum
#
#     def __repr__(self):
#         return 'ΣΣ{}'.format(repr(self.kernel))
#
#     @property
#     def theta(self):
#         return self.kernel.theta
#
#     def gencode(self, X, Y):
#         return ' + '.join([self.kernel.gencode(x, y) for x in X for y in Y])
