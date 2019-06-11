from graphdot.marginalized.basekernel import Constant
from graphdot.marginalized.basekernel import KroneckerDelta
from graphdot.marginalized.basekernel import SquareExponential

import random
import pytest

kernels = [
    Constant(1.0),
    KroneckerDelta(0.5),
    SquareExponential(1.0)
]


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


@pytest.mark.parametrize('kernel1', kernels)
@pytest.mark.parametrize('kernel2', kernels)
def test_kernel_add_kernel(kernel1, kernel2):
    ''' check by definition '''
    kadd = kernel1 + kernel2
    random.seed(0)
    for _ in range(1000):
        i = random.paretovariate(0.1)
        j = random.paretovariate(0.1)
        assert(kadd(i, j) == kernel1(i, j) + kernel2(i, j))
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


@pytest.mark.parametrize('kernel1', kernels)
@pytest.mark.parametrize('kernel2', kernels)
def test_kernel_mul_kernel(kernel1, kernel2):
    ''' check by definition '''
    kmul = kernel1 * kernel2
    random.seed(0)
    for _ in range(1000):
        i = random.paretovariate(0.1)
        j = random.paretovariate(0.1)
        assert(kmul(i, j) == kernel1(i, j) * kernel2(i, j))
        assert(kmul(i, j) == kmul(j, i))
