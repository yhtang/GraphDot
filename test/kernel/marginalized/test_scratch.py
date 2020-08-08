#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pytest
import pycuda.autoinit
from graphdot.kernel.marginalized._scratch import PCGScratch


@pytest.mark.parametrize('cls', [
    lambda n: PCGScratch(n),
])
def test_scratch(cls):
    assert(cls(1).dtype.isalignedstruct)

    with pytest.raises(ValueError):
        cls(0)

    with pytest.raises(ValueError):
        cls(-1)


@pytest.mark.parametrize('cls', [
    PCGScratch,
])
@pytest.mark.parametrize('size', [1, 11, 16, 17, 25, 31, 32, 217, 8195, 91924])
@pytest.mark.parametrize('ntemp', [1, 5, 6, 10])
def test_scratch_allocation(cls, size, ntemp):
    scratch = cls(size, ntemp)
    assert(scratch.nmax >= size)
    assert(scratch.nmax * scratch.ncol >= size * ntemp)
    assert(scratch.ptr != 0)
    assert(scratch.state)
    with pytest.raises(AttributeError):
        scratch.ptr = np.uint64(0)
    with pytest.raises(AttributeError):
        scratch.ptr = 0
