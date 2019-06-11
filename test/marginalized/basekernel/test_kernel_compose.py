from graphdot.marginalized.basekernel import Constant
from graphdot.marginalized.basekernel import KroneckerDelta
from graphdot.marginalized.basekernel import SquareExponential
from graphdot.marginalized.basekernel import Convolution
from graphdot.marginalized.basekernel import TensorProduct

import random
import pytest

inf = float('inf')
kernels = [
    Constant(1.0),
    KroneckerDelta(0.5),
    SquareExponential(1.0)
]


@pytest.mark.parametrize('kernel1', kernels)
@pytest.mark.parametrize('kernel2', kernels)
def test_tensor_product_2(kernel1, kernel2):
    k = TensorProduct(kernel1, kernel2)
    for i1, j1 in [(0, 0), (0, 1), (-1, 1), (0, inf)]:
        for i2, j2 in [(0, 0), (0, 1), (-1, 1), (0, inf)]:
            ''' default and corner cases '''
            assert(k((i1, i2), (j1, j2)) == kernel1(i1, j1) * kernel2(i2, j2))
    for _ in range(10000):
        i1 = random.paretovariate(0.1)
        j1 = random.paretovariate(0.1)
        i2 = random.paretovariate(0.1)
        j2 = random.paretovariate(0.1)
        ''' check by definition '''
        assert(k((i1, i2), (j1, j2)) == kernel1(i1, j1) * kernel2(i2, j2))


@pytest.mark.parametrize('kernel1', kernels)
@pytest.mark.parametrize('kernel2', kernels)
@pytest.mark.parametrize('kernel3', kernels)
def test_tensor_product_3(kernel1, kernel2, kernel3):
    k = TensorProduct(kernel1, kernel2, kernel3)
    ''' default and corner cases only '''
    for i1, j1 in [(0, 0), (0, 1), (-1, 1), (0, inf)]:
        for i2, j2 in [(0, 0), (0, 1), (-1, 1), (0, inf)]:
            for i3, j3 in [(0, 0), (0, 1), (-1, 1), (0, inf)]:
                assert(k((i1, i2, i3), (j1, j2, j3)) ==
                       kernel1(i1, j1) * kernel2(i2, j2) * kernel3(i3, j3))


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
