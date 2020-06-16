#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from pycuda import gpuarray
from graphdot.codegen.cpptool import cpptype


@cpptype(p_buffer=np.uintp, capacity=np.int64)
class BlockScratch:
    def __init__(self, capacity):
        self.capacity = ((capacity + 15) // 16) * 16
        if capacity <= 0:
            raise ValueError('Scratch size must be greater than zero.')
        self.buffer = gpuarray.empty(int(self.capacity) * 5, np.float32)

    @property
    def p_buffer(self):
        return self.buffer.ptr
