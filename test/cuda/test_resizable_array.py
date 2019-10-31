#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest
import numpy as np
import pycuda.autoinit
from graphdot.cuda.resizable_array import ResizableArray


@pytest.mark.parametrize('allocator', ['managed', 'numpy'])
def test_resizable_array_init_empty(allocator):

    arr = ResizableArray(np.float32, allocator=allocator)

    assert(arr.data is None)
    assert(arr.capacity == 0)
    assert(len(arr) == 0)


@pytest.mark.parametrize('allocator', ['managed', 'numpy'])
@pytest.mark.parametrize("count", [1, 2, 3, 5, 8, 15, 412, 651825, 8182512])
def test_resizable_array_init(allocator, count):

    arr = ResizableArray(np.float32, count, allocator=allocator)

    assert(arr.data is not None)
    assert(len(arr) == count)
    assert(arr.capacity >= count)


@pytest.mark.parametrize('allocator', ['managed', 'numpy'])
@pytest.mark.parametrize("count", [1, 2, 3, 5, 8, 15, 412, 651825, 8182512])
def test_resizable_array_append(allocator, count):

    arr = ResizableArray(np.float32, allocator=allocator)

    for _ in range(count):
        arr.append(0.0)

    assert(len(arr) == count)
    assert(arr.capacity >= count)


@pytest.mark.parametrize('allocator', ['managed', 'numpy'])
def test_resizable_array_iadd(allocator):

    np.random.seed(0)
    N = np.minimum(8192, 1 + np.random.pareto(0.2, 10)).astype(np.int32)
    M = np.minimum(8192, 1 + np.random.pareto(0.2, 10)).astype(np.int32)

    for n in N:
        for m in M:

            arr = ResizableArray(np.float32, n, allocator=allocator)
            arr[:] = -1
            arr += range(m)

            assert(len(arr) == n + m)
            assert(arr.capacity >= n + m)

            for i in range(n):
                assert(arr[i] == -1)
            for i in range(m):
                assert(arr[n + i] == i)
                assert(arr[-(i + 1)] == m - (i + 1))


@pytest.mark.parametrize('allocator', ['managed', 'numpy'])
def test_resizable_array_resize(allocator):

    arr = ResizableArray(np.float64, allocator=allocator)

    np.random.seed(0)
    N = np.minimum(1048576, 1 + np.random.pareto(0.1, 100)).astype(np.int32)
    for size in N:
        arr.resize(size)
        assert(len(arr) == size)
        assert(arr.capacity >= size)
        x, y = np.random.rand(2)
        arr[0] = x
        if size > 1:
            arr[-1] = y
        assert(x == pytest.approx(arr[0]))
        if size > 1:
            assert(y == pytest.approx(arr[-1]))


@pytest.mark.parametrize('allocator', ['managed', 'numpy'])
def test_resizable_array_reseve(allocator):

    np.random.seed(0)
    N = np.minimum(1048576, 1 + np.random.pareto(0.1, 100)).astype(np.int32)
    M = np.minimum(1048576, 1 + np.random.pareto(0.1, 100)).astype(np.int32)

    for n in N:
        for m in M:
            arr = ResizableArray(np.int32, n, allocator=allocator)
            arr[-1] = 12345
            if m > n:
                arr.reserve(m)
            else:
                with pytest.warns(UserWarning):
                    arr.reserve(m)
            assert(arr[-1] == 12345)
            assert(len(arr) == n)
            assert(arr.capacity >= m)
            assert(arr.capacity >= len(arr))


@pytest.mark.parametrize('allocator', ['managed', 'numpy'])
@pytest.mark.parametrize("count", [1, 2, 3, 5, 8, 15, 412, 651825, 8182512])
def test_resizable_array_clear(allocator, count):

    arr = ResizableArray(np.int16, count, allocator=allocator)

    previous_capacity = arr.capacity
    arr.clear()
    assert(len(arr) == 0)
    assert(arr.capacity == previous_capacity)


@pytest.mark.parametrize('allocator', ['managed', 'numpy'])
@pytest.mark.parametrize("count", [1, 2, 3, 5, 8, 15, 412, 651825])
def test_resizable_array_get_set(allocator, count):

    arr = ResizableArray(np.int64, count, allocator=allocator)

    for i in range(count):
        arr[i] = count - i

    for i in range(count):
        assert(arr[i] == count - i)

    arr.reserve(count * 2)

    for i in range(count):
        assert(arr[i] == count - i)
