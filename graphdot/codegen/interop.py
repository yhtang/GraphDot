#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np


class CppType(type):
    pass


def cpptype(dtype):
    dtype = np.dtype(dtype, align=True)

    def decor(clss):
        class MetaClass(CppType):
            @property
            def dtype(cls):
                return dtype

            def __repr__(cls):
                return '@cpptype({})\n{}'.format(repr(dtype), repr(clss))

        class Class(clss, metaclass=MetaClass):
            @property
            def state(self):
                state = []
                for key, (field, _) in Class.dtype.fields.items():
                    if field.names is not None:
                        state.append(getattr(self, key).state)
                    else:
                        state.append(getattr(self, key))
                return tuple(state)

            # TODO: need a nicer one
            # def __repr__(self):

        return Class

    return decor


if __name__ == '__main__':

    @cpptype([('A', np.int32), ('B', np.float32)])
    class A:
        def __init__(self):
            self.A = 1
            self.B = 2

    print(A.dtype)
    a = A()
    print(a.pack())
    a.A = 4
    print(a.pack())

    @cpptype([('X', np.bool_), ('Y', A.dtype)])
    class B:
        def __init__(self):
            self.X = True
            self.Y = A()
    #
    # print(A.dtype)
    # print(A().pack())
    print(B.dtype)
    print(B().pack())
