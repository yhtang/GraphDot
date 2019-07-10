#!/usr/bin/env python
# -*- coding: utf-8 -*-
from six import with_metaclass
import numpy as np
from graphdot.codegen import Template

__all__ = ['cpptype', 'decltype', 'rowtype']


# def cpptype(dtype=[], **kwtype):  # only works with python>=3.6
def cpptype(dtype=[]):
    """
    cpptype is a class decorator that simplifies the translation of python
    objects to corresponding C++ structures.
    """
    # only works with python >= 3.6
    # dtype = np.dtype(dtype + list(kwtype.items()), align=True)
    dtype = np.dtype(dtype, align=True)

    def decor(clss):
        class CppType(type):
            @property
            def dtype(cls):
                """
                use name 'dtype' in accordance with numpy, pandas, pycuda
                """
                return dtype

            def __repr__(cls):
                return '@cpptype({})\n{}'.format(repr(dtype), repr(clss))

        # class Class(clss, metaclass=CppType):
        class Class(with_metaclass(CppType, clss)):
            @property
            def dtype(self):
                return Class.dtype

            @property
            def state(self):
                state = []
                # break for python < 3.6 since field ordered not preserved
                # for key, (field, _) in Class.dtype.fields.items():
                for key in Class.dtype.names:
                    field, _ = Class.dtype.fields[key]
                    if field.names is not None:
                        state.append(getattr(self, key).state)
                    # elif field.subdtype is not None:
                    #     subtype, subshape = field.subdtype
                    #     subary = np.array(getattr(self, key), dtype=subtype)
                    #     assert(subary.shape == subshape)
                    #     state.append(subary.tolist())
                    else:
                        state.append(field.type(getattr(self, key)))
                return tuple(state)

            def __setattr__(self, name, value):
                if name in Class.dtype.names:
                    t = Class.dtype.fields[name][0]
                    if np.dtype(type(value)).kind != t.kind:
                        raise ValueError(
                            "Cannot set attribute '{}' (C++ type {}) "
                            "with value {} of {}".format(name, t,
                                                         value, type(value)))
                # super().__setattr__(name, value)
                # for python2/3 compat
                super(Class, self).__setattr__(name, value)

            # TODO: need a nicer __repr__

        return Class

    return decor


def decltype(type, name=''):
    type = np.dtype(type, align=True)  # convert np.float32 etc. to dtype
    if type.names is not None:
        if len(type.names):
            return Template(r'struct ${cls}{${members;};}${name}').render(
                cls=name,
                name=name,
                members=[decltype(type.fields[v][0], v) for v in type.names])
        else:
            return 'constexpr static numpy_type::_empty {} {{}}'.format(name)
    # elif type.subdtype is not None:
    #     return Template(r'''${type} ${name}${dim}''').render(
    #         type=type.name, name=
    #     )
    else:
        return '{} {}'.format(str(type.name), name)


def rowtype(df, pack=True):
    if pack is True:
        order = np.argsort([-df.dtypes[key].itemsize for key in df.columns])
    else:
        order = np.arange(len(df.columns))
    packed_attributes = [df.columns[i] for i in order]
    packed_dtype = np.dtype([(key, df.dtypes[key].newbyteorder('='))
                             for key in packed_attributes], align=True)
    return packed_dtype
