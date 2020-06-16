#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pickle
import numpy as np
import pytest
from graphdot.minipandas.series import Series


@pytest.mark.parametrize('array', [
    [1, 2, 3],
    [1.0, 2, 3],
    [1, 2.0, True],
    [None, None],
    ['hello', 'world'],
    ['hello', 2, 3, -1],
    np.arange(10),
    range(10),
])
def test_series_creation(array):
    s = Series(array)
    assert(isinstance(s, np.ndarray))
    assert(isinstance(s, Series))
    assert(len(s) == len(array))
    assert(isinstance(repr(s), str))


@pytest.mark.parametrize('case', [
    (np.array(10), np.int, np.int),
    (np.empty(10, dtype=np.bool_), np.bool_, np.bool_),
    (np.linspace(0, 1, 10), np.float, np.float),
    (np.array([1, 2, 3], dtype=np.object), np.object, np.int),
    (np.array(['hello', 'world']), np.dtype('U5'), np.dtype('U5')),
    (np.array(['hello', 'world!']), np.dtype('U6'), np.dtype('U6')),
    (np.array([(1, 2), (3, 4, 5)], dtype=np.object), np.object, tuple),
    ([(1, 2), (3, 4)], np.object, tuple),
    ([(1, 2), (3, 4, 5)], np.object, tuple),
    ([1, 2, 3], np.int16, np.int16),
    ([1.0, 2, 3], np.float32, np.float32),
    (['abc', 'd', 'ef'], np.dtype('U3'), np.dtype('U3')),
    # ([1, 'd', False], np.object, None),
])
def test_series_dtype(case):
    array, t_dtype, t_concrete = case
    s = Series(array)
    assert(s.concrete_type == t_concrete)
    assert(s.dtype == t_dtype)


@pytest.mark.parametrize('array', [
    np.linspace(0, 1, 100),
    np.arange(1000),
    np.zeros(100, dtype=np.bool_),
    np.array(['hello', 'world!']),
    np.array(['hello', True, 1.0, -2]),
])
def test_series_pickle(array):
    s = Series(array)
    pck = pickle.dumps(s)
    mirror = pickle.loads(pck)
    assert(len(s) == len(mirror))
    assert(s.concrete_type == mirror.concrete_type)
    for _1, _2 in zip(s, mirror):
        assert(_1 == _2)
