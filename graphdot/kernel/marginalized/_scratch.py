#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from pycuda import gpuarray
from graphdot.codegen.cpptool import cpptype


class Scratch:

    def __init__(self, nrow, ncol, h_align, w_align, dtype):
        self.h_align = h_align = int(h_align)
        self.w_align = w_align = int(w_align)
        self.nrow = ((nrow + h_align - 1) // h_align) * h_align
        self.ncol = ((ncol + w_align - 1) // w_align) * w_align
        if self.nrow <= 0 or self.ncol <= 0:
            raise ValueError('Scratch size must be greater than zero.')
        self.data = gpuarray.empty(self.nrow * self.ncol, dtype)


@cpptype(ptr=np.uintp, nmax=np.uint64)
class PCGScratch(Scratch):

    def __init__(self, capacity, n_temporaries=5):
        super().__init__(capacity, n_temporaries, 16, 1, np.float32)

    @property
    def nmax(self):
        return self.nrow

    @property
    def ndim(self):
        return self.ncol

    @property
    def ptr(self):
        return self.data.ptr
