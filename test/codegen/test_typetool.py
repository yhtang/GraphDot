#!/usr/bin/env python
# -*- coding: utf-8 -*-
from re import search
from copy import copy
import numpy as np
import pytest
import exrex
from graphdot.codegen.typetool import cpptype, decltype, rowtype
from graphdot.minipandas import DataFrame

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
    (np.float64, 'float64'),
    ('S1', 'char [1]'),
    ('S2', 'char [2]'),
    ('S5', 'char [5]'),
    ('S10', 'char [10]'),
]


@pytest.mark.parametrize('case', decltype_cases)
def test_decltype(case):
    dtype, typestring = case
    assert(decltype(dtype).strip() == typestring)


def test_decltype_compose():
    assert(decltype(np.float32) in decltype(comp1))
    assert(decltype(np.int16) in decltype(comp1))
    assert(decltype(comp1, 'x') in decltype(comp2))


@pytest.mark.parametrize('element_type', [
    np.bool_, np.byte, np.int, np.float, np.float32, np.float64, np.intp
])
@pytest.mark.parametrize('size', [
    (1,), (8,), (1, 1), (2, 3), (3, 5, 8)
])
def test_decltype_array(element_type, size):
    assert(decltype(str(size)+np.dtype(element_type).name) ==
           decltype(element_type) + ''.join(["[%d]" % d for d in size]))


def test_decltype_empty():
    assert('empty' in decltype([]))
    # TODO: use cppyy to verify that empty fields have zero sizes


def test_decltype_order():
    ''' ensure output member order is the same as that in dtype '''
    np.random.seed(0)
    type_list = [np.bool_, np.int8, np.int16, np.int32, np.int64,
                 np.uint8, np.uint16, np.uint32, np.uint64,
                 np.float32, np.float64]
    for _ in range(1024):
        length = np.random.randint(1, 16)
        member_types = np.random.choice(type_list, length)
        member_names = []
        while len(member_names) < length:
            name = exrex.getone('[_a-zA-Z][_0-9a-zA-Z]*', 16)
            if name not in member_names:
                member_names.append(name)
        type = np.dtype(list(zip(member_names, member_types)))
        cstr = decltype(type)
        for prev, next in zip(type.names[:-1], type.names[1:]):
            cprev = '%s;' % decltype(type.fields[prev][0], prev)
            cnext = '%s;' % decltype(type.fields[next][0], next)
            assert(search(cprev, cstr).start() <=
                   search(cnext, cstr).start())


rowtype_cases = [
    # empty dataframe
    (DataFrame([]), np.dtype([], align=True)),
    # different types of columns
    (DataFrame({'A': [1.5, 42.0],
                'B': [-1, 32768],
                'C': [False, True]}),
     np.dtype([('A', np.float64),
               ('B', np.int64),
               ('C', np.bool_)], align=True)),
    # small-big-small layout optimization
    (DataFrame({'A': [True, False],
                'B': [-1, 1],
                'C': [False, True]}),
     np.dtype([('A', np.bool_),
               ('B', np.int64),
               ('C', np.bool_)], align=True)),
]


@pytest.mark.parametrize('case', rowtype_cases)
def test_rowtype(case):
    df, dtype = case
    # assert(rowtype(df, pack=False) == dtype)
    assert(rowtype(df, pack=False).itemsize >= rowtype(df, pack=True).itemsize)
