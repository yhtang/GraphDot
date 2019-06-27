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
    def theta(self):
        return self.k1.theta + self.k2.theta

    @property
    def _layout(self):
        return '[{}][{}]'.format(self.k1._layout,
                                 self.k2._layout)

    @property
    def _decltype(self):
        return '{ns}::{cls}<{a1},{a2}>'.format(ns=__cpp_namespace__,
                                               cls=self.cls,
                                               a1=self.k1._decltype,
                                               a2=self.k2._decltype)


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
    def theta(self):
        return [self.constant]

    @property
    def _layout(self):
        return '[f]'

    @property
    def _decltype(self):
        return '{ns}::{cls}'.format(ns=__cpp_namespace__,
                                    cls='constant')

    def gencode(self, x, y):
        return '{:f}f'.format(self.constant)


class Multiply(Kernel):
    def __init__(self):
        pass

    def __call__(self, i, j):
        return i * j

    def __repr__(self):
        return '*'

    @property
    def theta(self):
        return []

    def gencode(self, x, y):
        return '({} * {})'.format(x, y)


class KroneckerDelta(Kernel):

    def __init__(self, lo, hi=1.0):
        self.lo = lo
        self.hi = hi

    def __call__(self, i, j):
        return self.hi if i == j else self.lo

    def __repr__(self):
        return 'δ({}, {})'.format(self.hi, self.lo)

    @property
    def theta(self):
        return [self.lo, self.hi]

    @property
    def _layout(self):
        return '[ff]'

    @property
    def _decltype(self):
        return '{ns}::{cls}'.format(ns=__cpp_namespace__,
                                    cls='kronecker_delta')

    def gencode(self, x, y):
        return '({} == {} ? {:f}f : {:f}f)'.format(x, y, self.hi, self.lo)


class SquareExponential(Kernel):
    def __init__(self, length_scale):
        self.length_scale = length_scale

    def __call__(self, x1, x2):
        return numpy.exp(-0.5 * numpy.sum((x1 - x2)**2) / self.length_scale**2)

    def __repr__(self):
        return 'SqExp({})'.format(self.length_scale)

    @property
    def theta(self):
        return [self.length_scale]

    @property
    def _layout(self):
        return '[f]'

    @property
    def _decltype(self):
        return '{ns}::{cls}'.format(ns=__cpp_namespace__,
                                    cls='square_exponential')

    def gencode(self, x, y):
        return 'expf({:f}f * power({} - {}, 2))'.format(
            -0.5 / self.length_scale**2, x, y)


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
    def theta(self):
        return [a for k in self.kernels for a in k.theta]

    @property
    def _layout(self):
        return '[{}]'.format(''.join([k._layout for k in self.kernels]))

    @property
    def _decltype(self):
        kernels = 'std::tuple<{}>'.format(','.join([k._decltype
                                                    for k in self.kernels]))
        idx = ','.join([str(i) for i in range(len(self.kernels))])
        return '{ns}::{cls}<{kernels},{idx}>'.format(ns=__cpp_namespace__,
                                                     cls='tensor_product',
                                                     kernels=kernels,
                                                     idx=idx)

    def gencode(self, X, Y):
        return ' * '.join([k.gencode(x, y)
                           for k, x, y in zip(self.kernels, X, Y)])


class KeywordTensorProduct(Kernel):
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

    @property
    def theta(self):
        return [a for k in self.kw_kernels.values() for a in k.theta]

    def gencode(self, x, y):
        # return ' * '.join([k.gencode(X[key], Y[key])
        #                    for key, k in self.kw_kernels.items()])
        return ' * '.join([k.gencode('%s.%s' % (x, key),
                                     '%s.%s' % (y, key))
                           for key, k in self.kw_kernels.items()])


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
    def theta(self):
        return self.kernel.theta

    @property
    def _layout(self):
        return '[{}]'.format(self.kernel._layout)

    @property
    def _decltype(self):
        return '{ns}::{cls}<{arg}>'.format(ns=__cpp_namespace__,
                                           cls='convolution',
                                           arg=self.kernel._decltype)

    def gencode(self, X, Y):
        return ' + '.join([self.kernel.gencode(x, y) for x in X for y in Y])


# if __name__ == '__main__':
#
#     kernels = [
#         Constant(0.5),
#         KroneckerDelta(0.5, 1.0),
#         SquareExponential(2.0),
#     ]
#     for k in kernels:
#         print(k.gencode('x', 'y'))
#
#     print(TensorProduct(KroneckerDelta(0.3,1.0),
#           SquareExponential(1.0)).gencode(['a1', 'b1'], ['a2', 'b2']))
#
#     print(Convolution(KroneckerDelta(0.3,1.0)).gencode(['a1', 'b1'],
#                                                        ['a2', 'b2', 'c2']))
#     print(Convolution(SquareExponential(1.0)).gencode(['a1', 'b1'],
#                                                       ['a2', 'b2', 'c2']))
