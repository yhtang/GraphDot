#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from pycuda.gpuarray import empty
from graphdot.codegen.typetool import cpptype


@cpptype([('ptr', np.uintp), ('capacity', np.int64)])
class BlockScratch:
    def __init__(self, capacity):
        self.capacity = ((capacity + 15) // 16) * 16
        self.buffer = empty(self.capacity * 4, np.float32)

    @property
    def ptr(self):
        return self.buffer.ptr


if __name__ == '__main__':

    import pycuda.autoinit

    scratch = BlockScratch(1024)

    print(scratch.state)
