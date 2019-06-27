#!/usr/bin/env python
# -*- coding: utf-8 -*-
from graphdot.codegen import Template
import numpy as np


def cpptype(dtype):
    dtype = np.dtype(dtype, align=True)

    def decor(clss):
        class CppType(type):
            @property
            def dtype(cls):
                return dtype

            def __repr__(cls):
                return '@cpptype({})\n{}'.format(repr(dtype), repr(clss))

        class Class(clss, metaclass=CppType):
            @property
            def state(self):
                state = []
                for key, (field, _) in Class.dtype.fields.items():
                    if field.names is not None:
                        state.append(getattr(self, key).state)
                    else:  # fixed-length vector treated the same as scalars
                        state.append(getattr(self, key))
                return tuple(state)

            # TODO: need a nicer __repr__

        return Class

    return decor


def decltype(type):
    type = np.dtype(type, align=True)  # convert np.float32 etc. to dtype
    if type.names is not None:
        return Template(r'''struct{${members;};}''').render(
            members=['{} {}'.format(decltype(t), v)
                     for v, (t, offset) in type.fields.items()])
    # elif type.subdtype is not None:
    #     return Template(r'''${type} ${name}${dim}''').render(
    #         type=type.name, name=
    #     )
    else:
        return str(type.name)


def rowtype(df, pack=True):
    if pack is True:
        order = np.argsort([-df.dtypes[key].itemsize for key in df.columns])
    else:
        order = np.arange(len(df.columns))
    packed_attributes = [df.columns[i] for i in order]
    packed_dtype = np.dtype([(key, df.dtypes[key].newbyteorder('='))
                             for key in packed_attributes], align=True)
    return packed_dtype


if __name__ == '__main__':

    @cpptype([('A', np.int32), ('B', np.float32)])
    class A:
        def __init__(self):
            self.A = 1
            self.B = 2

    print(A.dtype)
    a = A()
    print(a.state)
    a.A = 4
    print(a.state)

    @cpptype([('X', np.bool_), ('Y', A.dtype)])
    class B:
        def __init__(self):
            self.X = True
            self.Y = A()
    #
    # print(A.dtype)
    # print(A().state)
    print(B.dtype)
    print(B().state)
