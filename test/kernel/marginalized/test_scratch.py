#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pytest
import pycuda.autoinit
from graphdot.kernel.marginalized.scratch import BlockScratch


def test_scratch():
    assert(BlockScratch.dtype.isalignedstruct)

    with pytest.raises(ValueError):
        BlockScratch(0)

    with pytest.raises(ValueError):
        BlockScratch(-1)


sizes = [1, 11, 16, 17, 25, 31, 32, 217, 8195, 91924]


@pytest.mark.parametrize('size', sizes)
def test_scratch_allocation(size):
    scratch = BlockScratch(size)
    assert(scratch.capacity >= size)
    assert(scratch.ptr != 0)
    assert(scratch.state)
    with pytest.raises(AttributeError):
        scratch.ptr = np.uint64(0)
    with pytest.raises(ValueError):
        scratch.ptr = 0
