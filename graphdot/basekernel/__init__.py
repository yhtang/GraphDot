import struct
import numpy

__all__ = ['Constant',
           'KroneckerDelta',
           'SquareExponential',
           'TensorProduct',
           'Convolution']

__cpp_namespace__ = 'graphdot::basekernel'


class Kernel:
    def __add__(self, b):
        return KernelSum(self, b if isinstance(b, Kernel) else Constant(b))

    def __radd__(self, b):
        return KernelSum(b if isinstance(b, Kernel) else Constant(b), self)

    def __mul__(self, b):
        return KernelProd(self, b if isinstance(b, Kernel) else Constant(b))


class KernelOperator:
    def __init__(self, k1, k2):
        self.k1 = k1
        self.k2 = k2

    def __repr__(self):
        return '{} {} {}'.format(repr(self.k1), self.op, repr(self.k2))

    def __theta__(self):
        return self.k1.__theta__() + self.k2.__theta__()

    def __layout__(self):
        return '{{{}}}{{{}}}'.format(self.k1.__layout__(),
                                     self.k2.__layout__())

    def __decltype__(self):
        return '{ns}::{cls}<{a1},{a2}>'.format(ns=__cpp_namespace__,
                                               cls=self.cls,
                                               a1=self.k1.__decltype__(),
                                               a2=self.k2.__decltype__())


class KernelSum(KernelOperator):
    op = '+'
    cls = 'add'

    def __call__(self, object):
        return self.k1(object) + self.k2(object)


class KernelProd(KernelOperator):
    op = '*'
    cls = 'mul'

    def __call__(self, object):
        return self.k1(object) * self.k2(object)


class Constant(Kernel):
    def __init__(self, constant):
        self.constant = constant

    def __call__(self, object1, object2):
        return self.constant

    def __repr__(self):
        return '{}'.format(self.constant)

    def __theta__(self):
        return [self.constant]

    def __layout__(self):
        return '{f}'

    def __decltype__(self):
        return '{ns}::{cls}'.format(ns=__cpp_namespace__,
                                    cls='constant')


class KroneckerDelta(Kernel):

    def __init__(self, lo, hi=1.0):
        self.lo = lo
        self.hi = hi

    def __call__(self, object1, object2):
        return self.hi if object1 == object2 else self.lo

    def __repr__(self):
        return 'δ({}, {})'.format(self.hi, self.lo)

    def __theta__(self):
        return [self.lo, self.hi]

    def __layout__(self):
        return '{ff}'

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

    def __theta__(self):
        return [self.length_scale]

    def __layout__(self):
        return '{f}'

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

    def __theta__(self):
        return [a for k in self.kernels for a in k.__theta__()]

    def __layout__(self):
        return '{{{}}}'.format(''.join([k.__layout__() for k in self.kernels]))

    def __decltype__(self):
        arg = ','.join([k.__decltype__() for k in self.kernels])
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

    def __theta__(self):
        return self.kernel.__theta__()

    def __layout__(self):
        return '{{{}}}'.format(self.kernel.__layout__())

    def __decltype__(self):
        return '{ns}::{cls}<{arg}>'.format(ns=__cpp_namespace__,
                                           cls='convolution',
                                           arg=self.kernel.__decltype__())

if __name__ == '__main__':
    k1 = KroneckerDelta(1, 0.5)
    k2 = KroneckerDelta(0.9, 0.4)
    k3 = Constant(1.0)
    k4 = SquareExponential(1.0)
    print(k1)
    print(k2)
    print(k3)
    print(k4)
    print(TensorProduct(k1, k2))
    print(TensorProduct(k1, k2, k3))
    print(TensorProduct(k1, k2, k4))
    print(Convolution(k4))
    print(Convolution(k3))
    print(Convolution(k2))


    def examine(k):
        print('==============================')
        print(repr(k))
        print(k.__theta__())
        print(k.__layout__())
        print(k.__decltype__())
        print('==============================')


    examine(TensorProduct(k1, k2))
    examine(TensorProduct(k1, k3))
    examine(k1 + k3)
    examine(k2 + k4)
    examine(k4 + 1.0)
    examine(2.0 + k4)
    examine(k2 * k4)
    examine(k4 * 1.0)
    examine(Convolution(k4))
