#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from graphdot.codegen import Template

__all__ = ['cpptype', 'decltype', 'rowtype']


_convertible = {
    # lvalue : rvalue
    'b': 'b',
    'i': 'iu',
    'u': 'iu',
    'f': 'f',
    'c': 'c',
    'm': 'm',
    'M': 'M',
    'O': 'O',
    'S': 'S',
    'U': 'U',
    'V': 'V'
}


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

        class Class(clss, metaclass=CppType):
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
                    if np.dtype(type(value)).kind not in _convertible[t.kind]:
                        raise ValueError(
                            # f"Cannot set attribute '{name}' (C++ type {t}) "
                            # f"with value {value} of {type(value)}"
                            "Cannot set attribute '{name}' (C++ type {t}) "
                            "with value {value} of {type}".format(
                                name=name,
                                t=t,
                                value=value,
                                type=type(value)
                            )
                        )
                super().__setattr__(name, value)

            # TODO: need a nicer __repr__

        return Class

    return decor


def decltype(type, name=''):
    type = np.dtype(type, align=True)  # convert np.float32 etc. to dtype
    if type.names is not None:
        if len(type.names):
            return Template(r'struct{${members;};}${name}').render(
                name=name,
                members=[decltype(type.fields[v][0], v) for v in type.names])
        else:
            # return f'constexpr static _empty {name} {{}}'
            return 'constexpr static _empty {} {{}}'.format(name)
    # elif type.subdtype is not None:
    #     return Template(r'''${type} ${name}${dim}''').render(
    #         type=type.name, name=
    #     )
    else:
        # return f'{str(type.name)} {name}'
        return '{} {}'.format(str(type.name), name)


def rowtype(df, pack=True, exclude=[]):
    selection = np.array([key for key in df.columns if key not in exclude])
    if pack is True:
        perm = np.argsort([-df[key].dtype.itemsize for key in selection])
        selection = selection[perm]
    packed_dtype = np.dtype([(key, df[key].dtype.newbyteorder('='))
                             for key in selection], align=True)
    return packed_dtype
