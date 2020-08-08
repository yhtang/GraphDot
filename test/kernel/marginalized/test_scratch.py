#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pytest
import pycuda.autoinit
from graphdot.kernel.marginalized._scratch import PCGScratch


def test_scratch():
    assert(PCGScratch.dtype.isalignedstruct)

    with pytest.raises(ValueError):
        PCGScratch(0)

    with pytest.raises(ValueError):
        PCGScratch(-1)


sizes = [1, 11, 16, 17, 25, 31, 32, 217, 8195, 91924]


@pytest.mark.parametrize('size', sizes)
def test_scratch_allocation(size):
    scratch = PCGScratch(size)
    assert(scratch.capacity >= size)
    assert(scratch.p_buffer != 0)
    assert(scratch.state)
    with pytest.raises(AttributeError):
        scratch.p_buffer = np.uint64(0)
    with pytest.raises(AttributeError):
        scratch.p_buffer = 0
