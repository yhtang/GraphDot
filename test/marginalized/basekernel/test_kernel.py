from graphdot.basekernel import *

import pytest

kernels = [
    Constant(1.0),
    KroneckerDelta(0.5),
    SquareExponential(1.0)
]


@pytest.mark.parametrize('kernel', kernels)
def test_simple_kernels(kernel):
    assert(kernel(0, 0) == 1)
