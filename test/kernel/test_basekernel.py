#!/usr/bin/env python
# -*- coding: utf-8 -*-
import random
import copy
import numpy as np
import pytest

from graphdot.kernel.basekernel import Constant
from graphdot.kernel.basekernel import KroneckerDelta
from graphdot.kernel.basekernel import SquareExponential
from graphdot.kernel.basekernel import _Multiply
from graphdot.kernel.basekernel import TensorProduct
# from graphdot.marginalized.basekernel import Convolution

inf = float('inf')
nan = float('nan')
kernels = [
    Constant(1.0),
    # Multiply(),
    KroneckerDelta(0.5),
    SquareExponential(1.0)
]


@pytest.mark.parametrize('kernel', kernels)
def test_simple_kernel(kernel):
    ''' default behavior '''
    assert(kernel(0, 0) == 1)
    ''' corner cases '''
    assert(kernel(0, inf) <= 1)
    assert(kernel(0, -inf) <= 1)
    assert(kernel(inf, 0) <= 1)
    assert(kernel(-inf, 0) <= 1)
    ''' random input '''
    random.seed(0)
    for _ in range(10000):
        i = random.paretovariate(0.1)
        j = random.paretovariate(0.1)
        assert(kernel(i, j) >= 0 and kernel(i, j) <= 1)
        assert(kernel(i, j) == kernel(j, i))  # check symmetry
    ''' representation meaningness '''
    assert(eval(repr(kernel)).theta == kernel.theta)
    ''' hyperparameter retrieval '''
    assert(isinstance(kernel.theta, tuple))
    assert(len(kernel.theta) > 0)
    kernel.theta = kernel.theta
    another = copy.copy(kernel)
    for t1, t2 in zip(kernel.theta, another.theta):
        assert(t1 == t2)


def test_constant_kernel():
    kernel = Constant(1.0)
    ''' default behavior '''
    assert(kernel(0, 0) == 1)
    ''' corner cases '''
    assert(kernel(inf, inf) == 1)
    assert(kernel(inf, -inf) == 1)
    assert(kernel(nan, nan) == 1)
    assert(kernel(None, 0) == 1)
    assert(kernel(None, None) == 1)
    assert(kernel(1.0, 'a') == 1)
    ''' C++ code generation '''
    assert(kernel.dtype.isalignedstruct)
    assert(isinstance(kernel.gen_constexpr('x', 'y'), str))


def test_kronecker_delta_kernel():
    kernel = KroneckerDelta(0.5, 1.0)
    ''' default behavior '''
    assert(kernel(0, 0) == 1)
    assert(kernel('a', 'a') == 1.0)
    assert(kernel('a', 'b') == 0.5)
    ''' corner cases '''
    assert(kernel(inf, inf) == 1)
    assert(kernel(-inf, inf) == 0.5)
    assert(kernel(inf, -inf) == 0.5)
    assert(kernel(inf, -inf) == 0.5)
    assert(kernel(nan,  inf) == 0.5)
    assert(kernel(nan, -inf) == 0.5)
    assert(kernel(nan, nan) == 0.5)
    assert(kernel(None, 0) == 0.5)
    assert(kernel(None, None) == 1)
    assert(kernel(1.0, 'a') == 0.5)
    ''' C++ code generation '''
    assert(kernel.dtype.isalignedstruct)
    assert(isinstance(kernel.gen_constexpr('x', 'y'), str))


def test_square_exponential_kernel():
    kernel = SquareExponential(1.0)
    assert(kernel.theta.length_scale == pytest.approx(1.0))
    ''' default behavior '''
    assert(kernel(0, 0) == 1)
    assert(kernel(1, 1) == 1)
    assert(kernel(0, inf) == 0)
    ''' corner cases '''
    assert(kernel(-inf, inf) == 0)
    assert(kernel(inf, -inf) == 0)
    ''' C++ code generation '''
    assert(kernel.dtype.isalignedstruct)
    assert(isinstance(kernel.gen_constexpr('x', 'y'), str))


def test_multiply_quasikernel():
    kernel = _Multiply()
    ''' default behavior '''
    assert(kernel(0, 0) == 0)
    assert(kernel(0, 1) == 0)
    assert(kernel(1, 1) == 1)
    ''' random cases '''
    for r1, r2 in np.random.randn(1000, 2):
        assert(kernel(r1, r2) == r1 * r2)
    ''' C++ code generation '''
    assert(kernel.dtype.isalignedstruct)
    assert(isinstance(kernel.gen_constexpr('x', 'y'), str))
    ''' representation generation '''
    assert(isinstance(str(kernel), str))
    assert(isinstance(repr(kernel), str))
    assert(kernel.theta == tuple())
    kernel.theta = kernel.theta
    ''' representation meaningness '''
    assert(eval(repr(kernel)).theta == kernel.theta)


@pytest.mark.parametrize('k1', kernels)
@pytest.mark.parametrize('k2', kernels)
def test_tensor_product_2(k1, k2):
    k = TensorProduct(x=k1, y=k2)
    mirror = eval(repr(k))  # representation meaningness test
    assert(mirror.theta == k.theta)
    for i1, j1 in [(0, 0), (0, 1.5), (-1, 1), (-1.0, 0)]:
        for i2, j2 in [(0, 0), (0, 1.5), (-1, 1), (-1.0, 0)]:
            ''' default and corner cases '''
            assert(k(dict(x=i1, y=i2), dict(x=j1, y=j2))
                   == k1(i1, j1) * k2(i2, j2))
            assert(k(dict(x=i1, y=i2), dict(x=j1, y=j2))
                   == mirror(dict(x=i1, y=i2), dict(x=j1, y=j2)))
    for _ in range(10000):
        i1 = random.paretovariate(0.1)
        j1 = random.paretovariate(0.1)
        i2 = random.paretovariate(0.1)
        j2 = random.paretovariate(0.1)
        ''' check by definition '''
        assert(k(dict(x=i1, y=j1), dict(x=i2, y=j2))
               == k1(i1, i2) * k2(j1, j2))
        assert(k(dict(x=i1, y=i2), dict(x=j1, y=j2))
               == mirror(dict(x=i1, y=i2), dict(x=j1, y=j2)))
    ''' hyperparameter retrieval '''
    assert(k1.theta in k.theta)
    assert(k2.theta in k.theta)
    k.theta = k.theta
    ''' representation generation '''
    assert(len(str(k).split('⊗')) == 2)
    assert(str(k1) in str(k))
    assert(str(k2) in str(k))
    assert(repr(k1) in repr(k))
    assert(repr(k2) in repr(k))
    ''' C++ code generation '''
    assert(k.dtype.isalignedstruct)
    assert(isinstance(k.gen_constexpr('x', 'y'), str))


@pytest.mark.parametrize('k1', kernels)
@pytest.mark.parametrize('k2', kernels)
@pytest.mark.parametrize('k3', kernels)
def test_tensor_product_3(k1, k2, k3):
    k = TensorProduct(x=k1, y=k2, z=k3)
    mirror = eval(repr(k))  # representation meaningness test
    assert(mirror.theta == k.theta)
    ''' default and corner cases only '''
    for x1, y1, z1 in [(0, 0, 0), (0, 1, -1), (-1, 1, 0.5), (0, -42., 1)]:
        for x2, y2, z2 in [(0, 0, 0), (0, 1, -1), (-1, 1, 0.5), (0, -42., 1)]:
            ''' default and corner cases '''
            assert(k(dict(x=x1, y=y1, z=z1), dict(x=x2, y=y2, z=z2))
                   == (k1(x1, x2)
                   * k2(y1, y2)
                   * k3(z1, z2)))
            assert(k(dict(x=x1, y=y1, z=z1), dict(x=x2, y=y2, z=z2))
                   == mirror(dict(x=x1, y=y1, z=z1), dict(x=x2, y=y2, z=z2)))
    ''' hyperparameter retrieval '''
    assert(k1.theta in k.theta)
    assert(k2.theta in k.theta)
    assert(k3.theta in k.theta)
    k.theta = k.theta
    ''' representation generation '''
    assert(len(str(k).split('⊗')) == 3)
    assert(str(k1) in str(k))
    assert(str(k2) in str(k))
    assert(str(k3) in str(k))
    assert(repr(k1) in repr(k))
    assert(repr(k2) in repr(k))
    assert(repr(k3) in repr(k))
    ''' C++ code generation '''
    assert(k.dtype.isalignedstruct)
    assert(isinstance(k.gen_constexpr('x', 'y'), str))


# # @pytest.mark.parametrize('kernel', kernels)
# # def test_convolution(kernel):
# #     k = Convolution(kernel)
# #     ''' length cases '''
# #     assert(k([], []) == 0)
# #     assert(k(tuple(), tuple()) == 0)
# #     ''' check by definition '''
# #     for i, j in ([0, 0], [0, inf], [0, 1]):
# #         for length1 in range(10):
# #             for length2 in range(10):
# #                 assert(k([i] * length1, [j] * length2) ==
# #                        pytest.approx(kernel(i, j) * length1 * length2))
# #     ''' hyperparameter retrieval '''
# #     assert(len(k.theta) == len(kernel.theta))
# #     for t1, t2 in zip(k.theta, kernel.theta):
# #         assert(t1 == t2)
# #     ''' representation generation '''
# #     assert('ΣΣ' in repr(k))
# #     assert(repr(kernel) in repr(k))
# #     ''' C++ counterpart layout '''
# #     assert(kernel._layout in k._layout)
# #     ''' C++ counterpart type '''
# #     assert(kernel._decltype in k._decltype)
#
#
@pytest.mark.parametrize('kernel', kernels)
def test_kernel_add_constant(kernel):
    ''' check by definition '''
    for kadd in [kernel + 1, 1 + kernel]:
        random.seed(0)
        mirror = eval(repr(kadd))  # representation meaningness test
        assert(mirror.theta == kadd.theta)
        for _ in range(10000):
            i = random.paretovariate(0.1)
            j = random.paretovariate(0.1)
            assert(kadd(i, j) == kernel(i, j) + 1)
            assert(kadd(i, j) == kadd(j, i))
            assert(kadd(i, j) == mirror(i, j))
            assert(kadd(i, j) == mirror(j, i))
        ''' representation generation '''
        assert(len(str(kadd).split('+')) == 2)
        assert(str(kernel) in str(kadd))
        assert(len(repr(kadd).split('+')) == 2)
        assert(repr(kernel) in repr(kadd))
        ''' hyperparameter retrieval '''
        assert(kernel.theta in kadd.theta)
        kadd.theta = kadd.theta
        ''' C++ code generation '''
        assert(kadd.dtype.isalignedstruct)
        assert(isinstance(kadd.gen_constexpr('x', 'y'), str))


@pytest.mark.parametrize('k1', kernels)
@pytest.mark.parametrize('k2', kernels)
def test_kernel_add_kernel(k1, k2):
    ''' check by definition '''
    kadd = k1 + k2
    random.seed(0)
    mirror = eval(repr(kadd))  # representation meaningness test
    assert(mirror.theta == kadd.theta)
    for _ in range(1000):
        i = random.paretovariate(0.1)
        j = random.paretovariate(0.1)
        assert(kadd(i, j) == k1(i, j) + k2(i, j))
        assert(kadd(i, j) == kadd(j, i))
        assert(kadd(i, j) == mirror(i, j))
        assert(kadd(i, j) == mirror(j, i))
    ''' representation generation '''
    assert(len(str(kadd).split('+')) == 2)
    assert(str(k1) in str(kadd))
    assert(str(k2) in str(kadd))
    assert(len(repr(kadd).split('+')) == 2)
    assert(repr(k1) in repr(kadd))
    assert(repr(k2) in repr(kadd))
    ''' hyperparameter retrieval '''
    assert(k1.theta in kadd.theta)
    assert(k2.theta in kadd.theta)
    kadd.theta = kadd.theta
    ''' C++ code generation '''
    assert(kadd.dtype.isalignedstruct)
    assert(isinstance(kadd.gen_constexpr('x', 'y'), str))


@pytest.mark.parametrize('kernel', kernels)
def test_kernel_mul_constant(kernel):
    ''' check by definition '''
    for kmul in [kernel * 2, 2 * kernel]:
        random.seed(0)
        mirror = eval(repr(kmul))  # representation meaningness test
        assert(mirror.theta == kmul.theta)
        for _ in range(10000):
            i = random.paretovariate(0.1)
            j = random.paretovariate(0.1)
            assert(kmul(i, j) == kernel(i, j) * 2)
            assert(kmul(i, j) == kmul(j, i))
            assert(kmul(i, j) == mirror(i, j))
            assert(kmul(i, j) == mirror(j, i))
        ''' representation generation '''
        assert(len(str(kmul).split('*')) == 2)
        assert(str(kernel) in str(kmul))
        assert(len(repr(kmul).split('*')) == 2)
        assert(repr(kernel) in repr(kmul))
        ''' hyperparameter retrieval '''
        assert(kernel.theta in kmul.theta)
        kmul.theta = kmul.theta
        ''' C++ code generation '''
        assert(kmul.dtype.isalignedstruct)
        assert(isinstance(kmul.gen_constexpr('x', 'y'), str))


@pytest.mark.parametrize('k1', kernels)
@pytest.mark.parametrize('k2', kernels)
def test_kernel_mul_kernel(k1, k2):
    ''' check by definition '''
    kmul = k1 * k2
    random.seed(0)
    mirror = eval(repr(kmul))  # representation meaningness test
    assert(mirror.theta == kmul.theta)
    for _ in range(1000):
        i = random.paretovariate(0.1)
        j = random.paretovariate(0.1)
        assert(kmul(i, j) == k1(i, j) * k2(i, j))
        assert(kmul(i, j) == kmul(j, i))
        assert(kmul(i, j) == mirror(i, j))
        assert(kmul(i, j) == mirror(j, i))
    ''' representation generation '''
    assert(len(str(kmul).split('*')) == 2)
    assert(str(k1) in str(kmul))
    assert(str(k2) in str(kmul))
    assert(len(repr(kmul).split('*')) == 2)
    assert(repr(k1) in repr(kmul))
    assert(repr(k2) in repr(kmul))
    ''' hyperparameter retrieval '''
    assert(k1.theta in kmul.theta)
    assert(k2.theta in kmul.theta)
    kmul.theta = kmul.theta
    ''' C++ code generation '''
    assert(kmul.dtype.isalignedstruct)
    assert(isinstance(kmul.gen_constexpr('x', 'y'), str))
