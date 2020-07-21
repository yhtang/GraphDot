#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pytest
from graphdot.codegen.typetool import (
    can_cast,
    common_min_type,
    common_concrete_type,
    have_same_fields,
    _dtype_util
)


def test_can_cast():
    for t in [
        np.bool_, np.int, np.int8, np.int16, np.int32, np.uintp, np.intp,
        np.float32, np.float64, np.float, np.complex
    ]:
        assert(can_cast(t, t))

    assert(can_cast(np.int32, np.uint8))
    assert(can_cast(np.int32, np.uint64))
    assert(can_cast(np.uint32, np.int8))
    assert(can_cast(np.uint32, np.int64))
    assert(not can_cast(np.int, np.float))
    assert(not can_cast(np.float, np.int))


@pytest.mark.parametrize('values,options,outcome', [
    ([], dict(coerce=False, ensure_signed=False), None),
    ([], dict(coerce=False, ensure_signed=True), None),
    ([], dict(coerce=True, ensure_signed=False), None),
    ([], dict(coerce=True, ensure_signed=True), None),
    ([0, 1, 2, 3], dict(coerce=False, ensure_signed=False), np.uint8),
    ([0, 1, 2, 3], dict(coerce=False, ensure_signed=True), np.int16),
    ([0, 1, 2, 3], dict(coerce=True, ensure_signed=False), np.uint8),
    ([0, 1, 2, 3], dict(coerce=True, ensure_signed=True), np.int16),
    ([0, 1, 2, -1], dict(coerce=False, ensure_signed=False), None),
    ([0, 1, 2, -1], dict(coerce=False, ensure_signed=True), None),
    ([0, 1, 2, -1], dict(coerce=True, ensure_signed=False), np.int16),
    ([0, 1, 2, -1], dict(coerce=True, ensure_signed=True), np.int16),
    ([0, 1, 2.0, 3], dict(coerce=False, ensure_signed=False), None),
    ([0, 1, 2.0, 3], dict(coerce=False, ensure_signed=True), None),
    ([0, 1, 2.0, 3], dict(coerce=True, ensure_signed=False), np.float32),
    ([0, 1, 2.0, 3], dict(coerce=True, ensure_signed=True), np.float32),
    ([0, 1, 2.0, 3], dict(coerce=True, min_float=np.float16), np.float32),
    ([0, 1, 2e4, 3], dict(coerce=True, min_float=np.float16), np.float32),
    ([1.0, 2.0, 3.0], dict(coerce=True, min_float=np.float16), np.float16),
    ([1.0, 2.0, 3.0], dict(coerce=False, min_float=np.float16), np.float16),
    ([1.0, 2.0, 3e4], dict(coerce=True, min_float=np.float16), np.float16),
    ([1.0, 2.0, 3e4], dict(coerce=False, min_float=np.float16), np.float16),
    ([1.0, 2.0, 3e15], dict(coerce=True, min_float=np.float16), np.float32),
    ([1.0, 2.0, 3e15], dict(coerce=False, min_float=np.float16), None),
    (['a', 'b', 'c'], dict(coerce=False), np.dtype('U1')),
    (['a', 'b', 'cc'], dict(coerce=False), None),
    (['a', 'b', 'cc'], dict(coerce=True), np.dtype('U2')),
    ([(1, 2), (3, ), (4, 5, 6)], dict(coerce=True), tuple),
    ([(1, 2), (3, ), (4, 5, 6)], dict(coerce=False), tuple),
    ([(1, 2), (3, ), [4, 5, 6]], dict(coerce=True), np.object),
    ([(1, 2), (3, ), [4, 5, 6]], dict(coerce=False), None),
])
def test_common_min_type_of_values(values, options, outcome):
    assert(common_min_type.of_values(values, **options) == outcome)


@pytest.mark.parametrize('values,options,outcome', [
    ([0, 1, 2, 3], dict(coerce=False, ensure_signed=False), int),
    ([0, 1, 2, 3], dict(coerce=False, ensure_signed=True), int),
    ([0, 1, 2, 3], dict(coerce=True, ensure_signed=False), int),
    ([0, 1, 2, 3], dict(coerce=True, ensure_signed=True), int),
    ([0, 1, 2, -1], dict(coerce=False, ensure_signed=False), int),
    ([0, 1, 2, -1], dict(coerce=False, ensure_signed=True), int),
    ([0, 1, 2, -1], dict(coerce=True, ensure_signed=False), int),
    ([0, 1, 2, -1], dict(coerce=True, ensure_signed=True), int),
    ([0, 1, 2.0, 3], dict(coerce=False, ensure_signed=False), None),
    ([0, 1, 2.0, 3], dict(coerce=False, ensure_signed=True), None),
    ([0, 1, 2.0, 3], dict(coerce=True, ensure_signed=False), np.float64),
    ([0, 1, 2.0, 3], dict(coerce=True, ensure_signed=True), np.float64),
    ([0, 1, 2.0, 3], dict(coerce=True, min_float=np.float16), np.float64),
    ([0, 1, 2e4, 3], dict(coerce=True, min_float=np.float16), np.float64),
    ([1.0, 2.0, 3.0], dict(coerce=True, min_float=np.float16), np.float),
    ([1.0, 2.0, 3.0], dict(coerce=False, min_float=np.float16), np.float),
    ([1.0, 2.0, 3e4], dict(coerce=True, min_float=np.float16), np.float),
    ([1.0, 2.0, 3e4], dict(coerce=False, min_float=np.float16), np.float),
    ([1.0, 2.0, 3e15], dict(coerce=True, min_float=np.float16), np.float),
    ([1.0, 2.0, 3e15], dict(coerce=False, min_float=np.float16), np.float),
    (['a', 'b', 'c'], dict(coerce=False), str),
    (['a', 'b', 'cc'], dict(coerce=False), str),
    (['a', 'b', 'cc'], dict(coerce=True), str),
    ([(1, 2), (3, ), (4, 5, 6)], dict(coerce=True), tuple),
    ([(1, 2), (3, ), (4, 5, 6)], dict(coerce=False), tuple),
    ([(1, 2), (3, ), [4, 5, 6]], dict(coerce=True), np.object),
    ([(1, 2), (3, ), [4, 5, 6]], dict(coerce=False), None),
])
def test_common_min_type_of_types(values, options, outcome):
    assert(common_min_type.of_types(map(type, values), **options) == outcome)
