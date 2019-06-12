from graphdot.marginalized.basekernel import Constant
from graphdot.marginalized.basekernel import KroneckerDelta
from graphdot.marginalized.basekernel import SquareExponential

import random
import pytest

inf = float('inf')
nan = float('nan')
kernels = [
    Constant(1.0),
    KroneckerDelta(0.5),
    SquareExponential(1.0)
]


@pytest.mark.parametrize('kernel', kernels)
def test_simple_kernel_behavior(kernel):
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
