#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pytest
import pycuda.autoinit
from graphdot.kernel.marginalized._scratch import PCGScratch, JacobianScratch


@pytest.mark.parametrize('cls', [
    lambda n: PCGScratch(n),
    lambda n: JacobianScratch(n, 4)
])
def test_scratch(cls):
    assert(cls(1).dtype.isalignedstruct)

    with pytest.raises(ValueError):
        cls(0)

    with pytest.raises(ValueError):
        cls(-1)


sizes = [1, 11, 16, 17, 25, 31, 32, 217, 8195, 91924]


@pytest.mark.parametrize('cls', [
    lambda n: PCGScratch(n),
    lambda n: JacobianScratch(n, 4)
])
@pytest.mark.parametrize('size', sizes)
def test_scratch_allocation(cls, size):
    scratch = cls(size)
    assert(scratch.nmax >= size)
    assert(scratch.ptr != 0)
    assert(scratch.state)
    with pytest.raises(AttributeError):
        scratch.ptr = np.uint64(0)
    with pytest.raises(AttributeError):
        scratch.ptr = 0
