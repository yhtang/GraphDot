#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
from copy import copy
import numpy as np
import pandas as pd
import pytest
from graphdot.codegen.typetool import cpptype, decltype, rowtype


cpptype_cases = [
    ([], tuple()),
    ()
]


def test_cpptype():

    @cpptype([])
    class Null(object):
        pass

    @cpptype([('x', np.int32), ('y', np.float32)])
    class A(object):
        pass

    @cpptype([('A', A.dtype), ('B', np.bool_)])
    class X(object):
        pass

    assert(Null().state == tuple())
    assert('cpptype' in repr(Null))
    assert(Null.dtype.isalignedstruct)
    assert(Null().dtype.isalignedstruct)

    a = A()
    a.x = 1
    with pytest.raises(ValueError):
        a.y = 2
    a.y = 1.5
    a.z = False
    assert(len(a.state) == 2)
    assert(a.state == (1, pytest.approx(1.5)))
    assert(A.dtype.isalignedstruct)
    assert(a.dtype.isalignedstruct)

    x = X()
    x.A = copy(a)
    x.A.x = 3
    x.A.y = -1.4
    with pytest.raises(ValueError):
        x.A = 1
    with pytest.raises(ValueError):
        x.A = 1.5
    with pytest.raises(ValueError):
        x.A = True
    with pytest.raises(ValueError):
        x.A = np.zeros(5)
    x.B = True
    assert(len(x.state) == 2)
    assert(len(x.state[0]) == 2)
    assert(x.state == ((3, pytest.approx(-1.4)), True))
    assert(X.dtype.isalignedstruct)
    assert(x.dtype.isalignedstruct)


# only works with python >= 3.6
# def test_cpptype_kwtype():
#
#     @cpptype()
#     class Null(object):
#         pass
#
#     @cpptype(x=np.int32, y=np.float32)
#     class A(object):
#         pass
#
#     @cpptype(A=A.dtype, B=np.bool_)
#     class X(object):
#         pass
#
#     assert(Null().state == tuple())
#     assert('cpptype' in repr(Null))
#     assert(Null.dtype.isalignedstruct)
#     assert(Null().dtype.isalignedstruct)
#
#     a = A()
#     a.x = 1
#     with pytest.raises(ValueError):
#         a.y = 2
#     a.y = 1.5
#     a.z = False
#     assert(len(a.state) == 2)
#     assert(a.state == (1, pytest.approx(1.5)))
#     assert(A.dtype.isalignedstruct)
#     assert(a.dtype.isalignedstruct)
#
#     x = X()
#     x.A = copy(a)
#     x.A.x = 3
#     x.A.y = -1.4
#     with pytest.raises(ValueError):
#         x.A = 1
#     with pytest.raises(ValueError):
#         x.A = 1.5
#     with pytest.raises(ValueError):
#         x.A = True
#     with pytest.raises(ValueError):
#         x.A = np.zeros(5)
#     x.B = True
#     assert(len(x.state) == 2)
#     assert(len(x.state[0]) == 2)
#     assert(x.state == ((3, pytest.approx(-1.4)), True))
#     assert(X.dtype.isalignedstruct)
#     assert(x.dtype.isalignedstruct)


comp1 = np.dtype([('x', np.float32), ('y', np.int16)])
comp2 = np.dtype([('x', comp1), ('y', np.bool_)])
decltype_cases = [
    (np.bool_, 'bool'),
    (np.uint16, 'uint16'),
    (np.int32, 'int32'),
    (np.float64, 'float64')
]


@pytest.mark.parametrize('case', decltype_cases)
def test_decltype(case):
    dtype, typestring = case
    assert(decltype(dtype).strip() == typestring)


def test_decltype_compose():
    assert(decltype(np.float32) in decltype(comp1))
    assert(decltype(np.int16) in decltype(comp1))
    assert(decltype(comp1, 'x') in decltype(comp2))


rowtype_cases = [
    # empty dataframe
    (pd.DataFrame([]), np.dtype([], align=True)),
    # different types of columns
    (pd.DataFrame([(1.5, -1, False),
                   (42.0, 32768, True)],
                  columns=['A', 'B', 'C']),
     np.dtype([('A', np.float64),
               ('B', np.int64),
               ('C', np.bool_)], align=True)),
    # small-big-small layout optimization
    (pd.DataFrame([(True, -1, False),
                   (False, 1, True)],
                  columns=['A', 'B', 'C']),
     np.dtype([('A', np.bool_),
               ('B', np.int64),
               ('C', np.bool_)], align=True)),
]


@pytest.mark.parametrize('case', rowtype_cases)
def test_rowtype(case):
    df, dtype = case
    assert(rowtype(df, pack=False) == dtype)
    assert(rowtype(df, pack=False).itemsize >= rowtype(df, pack=True).itemsize)
