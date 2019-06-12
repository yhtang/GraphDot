from graphdot.marginalized.basekernel import Constant
from graphdot.marginalized.basekernel import KroneckerDelta
from graphdot.marginalized.basekernel import SquareExponential
from graphdot.marginalized.basekernel import Convolution
from graphdot.marginalized.basekernel import TensorProduct

import pytest
import copy

kernels = [
    Constant(1.0),
    KroneckerDelta(0.5),
    SquareExponential(1.0)
]


@pytest.mark.parametrize('kernel', kernels)
def test_simple_kernel_theta(kernel):
    assert(isinstance(kernel.__theta__(), list))
    assert(len(kernel.__theta__()) > 0)
    another = copy.copy(kernel)
    for t1, t2 in zip(kernel.__theta__(), another.__theta__()):
        assert(t1 == t2)


@pytest.mark.parametrize('k1', kernels)
@pytest.mark.parametrize('k2', kernels)
def test_tensor_product_theta(k1, k2):
    k = TensorProduct(k1, k2)
    assert(len(k.__theta__()) ==
           len(k1.__theta__()) +
           len(k2.__theta__()))
    for t1, t2 in zip(k.__theta__(), k1.__theta__() + k2.__theta__()):
        assert(t1 == t2)


@pytest.mark.parametrize('kernel', kernels)
def test_convolution_theta(kernel):
    k = Convolution(kernel)
    assert(len(k.__theta__()) == len(kernel.__theta__()))
    for t1, t2 in zip(k.__theta__(), kernel.__theta__()):
        assert(t1 == t2)
