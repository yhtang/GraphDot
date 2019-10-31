#!/usr/bin/env python
# -*- coding: utf-8 -*-
import warnings
import numpy as np
from pycuda.driver import managed_empty
from pycuda.driver import mem_attach_flags as ma_flags


def managed_allocator(count, dtype):
    return managed_empty(count, dtype, 'C', ma_flags.GLOBAL)


def numpy_allocator(count, dtype):
    return np.empty(count, dtype)


class ResizableArray:
    """
    Python version of std::vector. To be used together with CUDA managed memory
    to reduce kernel launch delay.
    """

    def __init__(self, dtype, count=0, allocator='managed'):
        self.dtype = dtype
        self.allocator = self._choose_allocator(allocator)
        self._ptr = None
        self._active = None
        self._size = 0
        self._capacity = 0
        if count:
            self.resize(count)

    def _choose_allocator(self, allocator):
        if isinstance(allocator, str):
            if allocator == 'managed':
                return managed_allocator
            elif allocator == 'numpy':
                return numpy_allocator
            else:
                raise ValueError('Unknown allocator "%s"' % allocator)
        else:
            return allocator

    def append(self, value):
        if self._size == self._capacity:
            self.reserve(self._capacity * 2 + 1)
        self._ptr[self._size] = value
        self._size += 1
        self._update_active()

    def resize(self, count):
        count = int(count)
        if count > self._capacity:
            self.reserve(count)
        self._size = count
        self._update_active()

    def reserve(self, count):
        count = int(count)
        if count <= self._size:
            warnings.warn('Reserving no more than current size has no effect')
            return
        _new = self.allocator(count, self.dtype)
        if self._size > 0:  # copy data to new buffer
            _new[:self._size] = self._ptr[:self._size]
        self._ptr = _new
        self._capacity = count
        self._update_active()

    def clear(self):
        self._size = 0
        self._active = None

    def __len__(self):
        return self._size

    def __getitem__(self, key):
        return self._active[key]

    def __setitem__(self, key, value):
        self._active[key] = value

    def __iadd__(self, gen):
        lst = list(gen)
        ext_size = self._size + len(lst)
        self.reserve(ext_size)
        self._ptr[self._size:ext_size] = lst
        self._size = ext_size
        self._update_active()
        return self

    @property
    def data(self):
        return self._ptr

    @property
    def capacity(self):
        return self._capacity

    def _update_active(self):
        self._active = self._ptr[:self._size]
