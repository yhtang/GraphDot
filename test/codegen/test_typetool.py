#!/usr/bin/env python
# -*- coding: utf-8 -*-
from re import search
from copy import copy
import numpy as np
import pytest
import exrex
from graphdot.codegen.typetool import cpptype, decltype


class ListSpecs:
    @cpptype([])
    class Null(object):
        pass

    @cpptype([('x', np.int32), ('y', np.float32)])
    class A(object):
        pass

    @cpptype([('A', A.dtype), ('B', np.bool_)])
    class X(object):
        pass


class KWSpecs:
    @cpptype()
    class Null(object):
        pass

    @cpptype(x=np.int32, y=np.float32)
    class A(object):
        pass

    @cpptype(A=A.dtype, B=np.bool_)
    class X(object):
        pass


@pytest.mark.parametrize('Null', [ListSpecs.Null, KWSpecs.Null])
@pytest.mark.parametrize('A', [ListSpecs.A, KWSpecs.A])
@pytest.mark.parametrize('X', [ListSpecs.X, KWSpecs.X])
def test_cpptype(Null, A, X):

    assert(Null().state == tuple())
    assert('cpptype' in repr(Null))
    assert(Null.dtype.isalignedstruct)
    assert(Null().dtype.isalignedstruct)

    a = A()
    a.x = 1
    with pytest.raises(TypeError):
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
    with pytest.raises(TypeError):
        x.A = 1
    with pytest.raises(TypeError):
        x.A = 1.5
    with pytest.raises(TypeError):
        x.A = True
    with pytest.raises(TypeError):
        x.A = np.zeros(5)
    x.B = True
    assert(len(x.state) == 2)
    assert(len(x.state[0]) == 2)
    assert(x.state == ((3, pytest.approx(-1.4)), True))
    assert(X.dtype.isalignedstruct)
    assert(x.dtype.isalignedstruct)


def test_cpptype_array():

    @cpptype(w=(np.float32, 4))
    class Filter:
        pass

    f = Filter()

    f.w = np.zeros(4)
    f.w = [0.0, 1.0, 2.0, 3.0]
    with pytest.raises(ValueError):
        f.w = 1
    with pytest.raises(TypeError):
        f.w = np.zeros(4, dtype=np.bool_)
    with pytest.raises(TypeError):
        f.w = [1, 2, 3, 4]
    with pytest.raises(ValueError):
        f.w = np.zeros(8)

    assert(len(f.state) == 1)
    assert(len(f.state[0]) == 4)

    with pytest.raises(TypeError):
        @cpptype(w=(np.float32, 4))
        class Faulty:
            @property
            def w(self):
                return [1]

        Faulty().state

    with pytest.raises(ValueError):
        @cpptype(w=(np.float32, 4))
        class Faulty:
            @property
            def w(self):
                return np.ones(3)

        Faulty().state

    # test array-of-objects states
    @cpptype(a=np.int, b=np.bool_)
    class Object:
        pass

    @cpptype(array=(Object.dtype, 4))
    class ArrayOfObjects:
        pass

    obj = Object()
    aoo = ArrayOfObjects()
    obj.a = 1
    obj.b = False
    aoo.array = [obj, obj, obj, obj]
    assert(len(aoo.state) == 1)
    assert(len(aoo.state[0]) == 4)
    assert(len(aoo.state[0][0]) == 2)


@pytest.mark.parametrize('case', [
    (np.bool_, 'bool'),
    (np.uint16, 'uint16'),
    (np.int32, 'int32'),
    (np.float64, 'float64'),
])
def test_decltype_scalar(case):
    dtype, typestring = case
    assert(decltype(dtype).strip() == typestring)


@pytest.mark.parametrize('case', [
    ('S1', 'char [1]'),
    ('S2', 'char [2]'),
    ('S5', 'char [5]'),
    ('S10', 'char [10]'),
])
def test_decltype_string(case):
    dtype, typestring = case
    assert(decltype(dtype).strip() == typestring)


def test_decltype_compose():
    comp1 = np.dtype([('x', np.float32), ('y', np.int16)])
    comp2 = np.dtype([('x', comp1), ('y', np.bool_)])

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
    assert(decltype((element_type, size)) ==
           decltype(element_type) + ' ' + ''.join(["[%d]" % d for d in size]))
    assert(decltype(str(size)+np.dtype(element_type).name) ==
           decltype(element_type) + ' ' + ''.join(["[%d]" % d for d in size]))


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
