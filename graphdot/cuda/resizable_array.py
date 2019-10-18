#!/usr/bin/env python
# -*- coding: utf-8 -*-
import warnings
from pycuda.driver import managed_empty
from pycuda.driver import mem_attach_flags as ma_flags


class ResizableArray:
    def __init__(self, type, count=0, allocator=managed_empty):
        self.type = type
        self.allocator = allocator
        self._ptr = None
        self._active = None
        self._size = 0
        self._capacity = 0
        if count:
            self.resize(count)

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
        _new = self.allocator(count, self.type, 'C', ma_flags.GLOBAL)
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
        for value in gen:
            if self._size == self._capacity:
                self.reserve(self._capacity * 2 + 1)
            self._ptr[self._size] = value
            self._size += 1
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
