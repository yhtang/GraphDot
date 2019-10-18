#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest
import numpy as np
import pycuda.autoinit
from graphdot.cuda.resizable_array import ResizableArray

    
@pytest.mark.parametrize("dtype", [np.bool_, np.int16, np.float32, np.float64])
@pytest.mark.parametrize("count", [0, 1, 16, 256, 4096, 1048576])
def test_resizable_array_init(benchmark, dtype, count):

    def fun():
        arr = ResizableArray(dtype, count)

    benchmark.pedantic(fun, iterations=5, rounds=5, warmup_rounds=1)


def test_list_iadd_1M(benchmark):

    def fun():
        arr = []
        arr += range(1048576)

    benchmark.pedantic(fun, iterations=5, rounds=1, warmup_rounds=0)


@pytest.mark.parametrize("dtype", [np.bool_, np.int16, np.float32, np.float64])
def test_resizable_array_iadd_1M(benchmark, dtype):

    def fun():
        arr = ResizableArray(dtype)
        arr += range(1048576)

    benchmark.pedantic(fun, iterations=5, rounds=1, warmup_rounds=0)


@pytest.mark.parametrize("dtype", [np.bool_, np.int16, np.float32, np.float64])
def test_resizable_array_setitem_1M(benchmark, dtype):

    def fun():
        arr = ResizableArray(dtype, 1048576)
        arr[:] = range(1048576)

    benchmark.pedantic(fun, iterations=5, rounds=1, warmup_rounds=0)


@pytest.mark.parametrize("dtype", [np.bool_, np.int16, np.float32, np.float64])
def test_ndarray_setitem_1M(benchmark, dtype):

    def fun():
        arr = np.empty(1048576, dtype=dtype)
        arr[:] = range(1048576)

    benchmark.pedantic(fun, iterations=5, rounds=1, warmup_rounds=0)


@pytest.mark.parametrize("dtype", [np.bool_, np.int16, np.float32, np.float64])
def test_resizable_array_append_1M(benchmark, dtype):

    def fun():
        arr = ResizableArray(dtype)
        for i in range(1048576):
            arr.append(i)

    benchmark.pedantic(fun, iterations=5, rounds=1, warmup_rounds=0)
