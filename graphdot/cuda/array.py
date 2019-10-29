#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from pycuda.driver import (managed_empty, managed_empty_like, managed_zeros)
from pycuda.driver import mem_attach_flags as ma_flags


class ManagedArray(np.ndarray):
    @property
    def ptr(self):
        return int(self.base.base)


def umempty(size, dtype=np.float32):
    return managed_empty(size, dtype, 'C', ma_flags.GLOBAL)


def umzeros(size, dtype=np.float32):
    return managed_zeros(size, dtype, 'C', ma_flags.GLOBAL)


def umlike(array):
    u = managed_empty_like(array, ma_flags.GLOBAL)
    u[:] = array[:]
    return u


def umarray(array):
    u = managed_empty_like(array, ma_flags.GLOBAL)
    u[:] = array[:]
    return u
