#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from pycuda import gpuarray
from graphdot.codegen.typetool import cpptype


# only works with python >= 3.6
# @cpptype(ptr=np.uintp, capacity=np.int64)
@cpptype([('ptr', np.uintp), ('capacity', np.int64)])
class BlockScratch(object):
    def __init__(self, capacity):
        self.capacity = ((capacity + 15) // 16) * 16
        if capacity <= 0:
            raise ValueError('Scratch size must be greater than zero.')
        self.buffer = gpuarray.empty(self.capacity * 4, np.float32)

    @property
    def ptr(self):
        return self.buffer.ptr
