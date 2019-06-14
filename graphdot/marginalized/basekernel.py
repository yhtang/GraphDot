#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module defines base kernels and composibility rules for creating vertex
and edges kernels for the marginalized graph kernel.
"""
import numpy

__all__ = ['Constant',
           'KroneckerDelta',
           'SquareExponential',
           'TensorProduct',
           'Convolution']

__cpp_namespace__ = 'graphdot::basekernel'


class Kernel:
    """
    Parent class for all base kernels
    """

    def __add__(self, b):
        return KernelSum(self, b if isinstance(b, Kernel) else Constant(b))

    def __radd__(self, b):
        return KernelSum(b if isinstance(b, Kernel) else Constant(b), self)

    def __mul__(self, b):
        return KernelProd(self, b if isinstance(b, Kernel) else Constant(b))

    def __rmul__(self, b):
        return KernelProd(b if isinstance(b, Kernel) else Constant(b), self)


class KernelOperator:
    def __init__(self, k1, k2):
        self.k1 = k1
        self.k2 = k2

    def __repr__(self):
        return '{} {} {}'.format(repr(self.k1), self.op, repr(self.k2))

    @property
    def __theta__(self):
        return self.k1.__theta__ + self.k2.__theta__

    @property
    def __layout__(self):
        return '[{}][{}]'.format(self.k1.__layout__,
                                 self.k2.__layout__)

    @property
    def __decltype__(self):
        return '{ns}::{cls}<{a1},{a2}>'.format(ns=__cpp_namespace__,
                                               cls=self.cls,
                                               a1=self.k1.__decltype__,
                                               a2=self.k2.__decltype__)


class KernelSum(KernelOperator):
    op = '+'
    cls = 'add'

    def __call__(self, i, j):
        return self.k1(i, j) + self.k2(i, j)


class KernelProd(KernelOperator):
    op = '*'
    cls = 'mul'

    def __call__(self, i, j):
        return self.k1(i, j) * self.k2(i, j)


class Constant(Kernel):
    def __init__(self, constant):
        self.constant = constant

    def __call__(self, i, j):
        return self.constant

    def __repr__(self):
        return '{}'.format(self.constant)

    @property
    def __theta__(self):
        return [self.constant]

    @property
    def __layout__(self):
        return '[f]'

    @property
    def __decltype__(self):
        return '{ns}::{cls}'.format(ns=__cpp_namespace__,
                                    cls='constant')


class KroneckerDelta(Kernel):

    def __init__(self, lo, hi=1.0):
        self.lo = lo
        self.hi = hi

    def __call__(self, i, j):
        return self.hi if i == j else self.lo

    def __repr__(self):
        return 'δ({}, {})'.format(self.hi, self.lo)

    @property
    def __theta__(self):
        return [self.lo, self.hi]

    @property
    def __layout__(self):
        return '[ff]'

    @property
    def __decltype__(self):
        return '{ns}::{cls}'.format(ns=__cpp_namespace__,
                                    cls='kronecker_delta')


class SquareExponential(Kernel):
    def __init__(self, length_scale):
        self.length_scale = length_scale

    def __call__(self, x1, x2):
        return numpy.exp(-0.5 * numpy.sum((x1 - x2)**2) / self.length_scale**2)

    def __repr__(self):
        return 'SqExp({})'.format(self.length_scale)

    @property
    def __theta__(self):
        return [self.length_scale]

    @property
    def __layout__(self):
        return '[f]'

    @property
    def __decltype__(self):
        return '{ns}::{cls}'.format(ns=__cpp_namespace__,
                                    cls='square_exponential')


class TensorProduct(Kernel):
    def __init__(self, *kernels):
        self.kernels = kernels

    def __call__(self, object1, object2):
        prod = 1.0
        for kernel, part1, part2 in zip(self.kernels, object1, object2):
            prod *= kernel(part1, part2)
        return prod

    def __repr__(self):
        return ' ⊗ '.join([repr(k) for k in self.kernels])

    @property
    def __theta__(self):
        return [a for k in self.kernels for a in k.__theta__]

    @property
    def __layout__(self):
        return '[{}]'.format(''.join([k.__layout__ for k in self.kernels]))

    @property
    def __decltype__(self):
        arg = ','.join([k.__decltype__ for k in self.kernels])
        return '{ns}::{cls}<{arg}>'.format(ns=__cpp_namespace__,
                                           cls='tensor_product',
                                           arg=arg)


class Convolution(Kernel):
    def __init__(self, kernel):
        self.kernel = kernel

    def __call__(self, object1, object2):
        sum = 0.0
        for part1 in object1:
            for part2 in object2:
                sum += self.kernel(part1, part2)
        return sum

    def __repr__(self):
        return 'ΣΣ{}'.format(repr(self.kernel))

    @property
    def __theta__(self):
        return self.kernel.__theta__

    @property
    def __layout__(self):
        return '[{}]'.format(self.kernel.__layout__)

    @property
    def __decltype__(self):
        return '{ns}::{cls}<{arg}>'.format(ns=__cpp_namespace__,
                                           cls='convolution',
                                           arg=self.kernel.__decltype__)
