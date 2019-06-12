from graphdot.marginalized.basekernel import Constant
from graphdot.marginalized.basekernel import KroneckerDelta
from graphdot.marginalized.basekernel import SquareExponential
from graphdot.marginalized.basekernel import Convolution
from graphdot.marginalized.basekernel import TensorProduct

import random
import copy
import pytest

inf = float('inf')
nan = float('nan')
kernels = [
    Constant(1.0),
    KroneckerDelta(0.5),
    SquareExponential(1.0)
]


@pytest.mark.parametrize('kernel', kernels)
def test_simple_kernel(kernel):
    ''' default behavior '''
    assert(kernel(0, 0) == 1)
    assert(isinstance(repr(kernel), str))
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
    ''' hyperparameter retrieval '''
    assert(isinstance(kernel.__theta__(), list))
    assert(len(kernel.__theta__()) > 0)
    another = copy.copy(kernel)
    for t1, t2 in zip(kernel.__theta__(), another.__theta__()):
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


@pytest.mark.parametrize('k1', kernels)
@pytest.mark.parametrize('k2', kernels)
def test_tensor_product_2(k1, k2):
    k = TensorProduct(k1, k2)
    assert(isinstance(repr(k), str))
    for i1, j1 in [(0, 0), (0, 1), (-1, 1), (0, inf)]:
        for i2, j2 in [(0, 0), (0, 1), (-1, 1), (0, inf)]:
            ''' default and corner cases '''
            assert(k((i1, i2), (j1, j2)) == k1(i1, j1) * k2(i2, j2))
    for _ in range(10000):
        i1 = random.paretovariate(0.1)
        j1 = random.paretovariate(0.1)
        i2 = random.paretovariate(0.1)
        j2 = random.paretovariate(0.1)
        ''' check by definition '''
        assert(k((i1, i2), (j1, j2)) == k1(i1, j1) * k2(i2, j2))
    ''' hyperparameter retrieval '''
    k = TensorProduct(k1, k2)
    assert(len(k.__theta__()) ==
           len(k1.__theta__()) +
           len(k2.__theta__()))
    for t1, t2 in zip(k.__theta__(), k1.__theta__() + k2.__theta__()):
        assert(t1 == t2)


@pytest.mark.parametrize('k1', kernels)
@pytest.mark.parametrize('k2', kernels)
@pytest.mark.parametrize('k3', kernels)
def test_tensor_product_3(k1, k2, k3):
    k = TensorProduct(k1, k2, k3)
    ''' default and corner cases only '''
    for i1, j1 in [(0, 0), (0, 1), (-1, 1), (0, inf)]:
        for i2, j2 in [(0, 0), (0, 1), (-1, 1), (0, inf)]:
            for i3, j3 in [(0, 0), (0, 1), (-1, 1), (0, inf)]:
                assert(k((i1, i2, i3), (j1, j2, j3)) ==
                       k1(i1, j1) * k2(i2, j2) * k3(i3, j3))
    ''' hyperparameter retrieval '''
    k = TensorProduct(k1, k2, k3)
    assert(len(k.__theta__()) ==
           len(k1.__theta__()) +
           len(k2.__theta__()) +
           len(k3.__theta__()))
    for t1, t2 in zip(k.__theta__(),
                      k1.__theta__() +
                      k2.__theta__() +
                      k3.__theta__()):
        assert(t1 == t2)


@pytest.mark.parametrize('kernel', kernels)
def test_convolution(kernel):
    k = Convolution(kernel)
    ''' length cases '''
    assert(k([], []) == 0)
    assert(k(tuple(), tuple()) == 0)
    ''' check by definition '''
    for i, j in ([0, 0], [0, inf], [0, 1]):
        for length1 in range(10):
            for length2 in range(10):
                assert(k([i] * length1, [j] * length2) ==
                       pytest.approx(kernel(i, j) * length1 * length2))
    ''' hyperparameter retrieval '''
    k = Convolution(kernel)
    assert(len(k.__theta__()) == len(kernel.__theta__()))
    for t1, t2 in zip(k.__theta__(), kernel.__theta__()):
        assert(t1 == t2)


@pytest.mark.parametrize('kernel', kernels)
def test_kernel_add_constant(kernel):
    ''' check by definition '''
    kadd = kernel + 1
    random.seed(0)
    for _ in range(10000):
        i = random.paretovariate(0.1)
        j = random.paretovariate(0.1)
        assert(kadd(i, j) == kernel(i, j) + 1)
        assert(kadd(i, j) == kadd(j, i))


@pytest.mark.parametrize('k1', kernels)
@pytest.mark.parametrize('k2', kernels)
def test_kernel_add_kernel(k1, k2):
    ''' check by definition '''
    kadd = k1 + k2
    random.seed(0)
    for _ in range(1000):
        i = random.paretovariate(0.1)
        j = random.paretovariate(0.1)
        assert(kadd(i, j) == k1(i, j) + k2(i, j))
        assert(kadd(i, j) == kadd(j, i))


@pytest.mark.parametrize('kernel', kernels)
def test_kernel_mul_constant(kernel):
    ''' check by definition '''
    kmul = kernel * 2
    random.seed(0)
    for _ in range(10000):
        i = random.paretovariate(0.1)
        j = random.paretovariate(0.1)
        assert(kmul(i, j) == kernel(i, j) * 2)
        assert(kmul(i, j) == kmul(j, i))


@pytest.mark.parametrize('k1', kernels)
@pytest.mark.parametrize('k2', kernels)
def test_kernel_mul_kernel(k1, k2):
    ''' check by definition '''
    kmul = k1 * k2
    random.seed(0)
    for _ in range(1000):
        i = random.paretovariate(0.1)
        j = random.paretovariate(0.1)
        assert(kmul(i, j) == k1(i, j) * k2(i, j))
        assert(kmul(i, j) == kmul(j, i))
