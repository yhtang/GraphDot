#!/usr/bin/env python
# -*- coding: utf-8 -*-
import itertools as it
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


def can_cast(src, dst):
    return src.kind in _convertible[dst.kind]


class _dtype_util:
    @staticmethod
    def is_object(t):
        return t.names is not None

    @staticmethod
    def is_array(t):
        return t.subdtype is not None


def cpptype(dtype=[], **kwtypes):  # kwtypes only works with python>=3.6
    """
    cpptype is a class decorator that simplifies the translation of python
    objects to corresponding C++ structures.
    """
    dtype = np.dtype(dtype + list(kwtypes.items()), align=True)

    def decor(clss):
        class CppType(type):
            @property
            def dtype(cls):
                """
                use name 'dtype' in accordance with numpy, pandas, pycuda
                """
                return dtype

            def __repr__(cls):
                return f'@cpptype({repr(dtype)})\n{repr(clss)}'

        class Class(clss, metaclass=CppType):
            @property
            def dtype(self):
                return Class.dtype

            @property
            def state(self):
                state = []
                for key, (field, _) in Class.dtype.fields.items():
                    if _dtype_util.is_object(field):
                        state.append(getattr(self, key).state)
                    elif _dtype_util.is_array(field):
                        states = getattr(self, key)
                        if not isinstance(states, np.ndarray):
                            raise TypeError(
                                f'attribute {key} declared as an array'
                                f'but the actual type is not a numpy'
                                f'ndarray.'
                            )
                        if states.shape != field.shape:
                            raise ValueError(
                                f'attribute {key}: '
                                f'actual shape of {states.shape} does not '
                                f'match declared shape of {field.shape}.'
                            )
                        if _dtype_util.is_object(field.base):
                            states = [
                                states[coord].state for coord
                                in it.product(*map(range, field.shape))
                            ]
                        state.append(
                            np.array(states, dtype=field.base)
                              .reshape(field.shape).tolist())
                    else:
                        state.append(field.type(getattr(self, key)))
                return tuple(state)

            def __setattr__(self, name, value):
                if name in Class.dtype.names:
                    lt = Class.dtype.fields[name][0]
                    if _dtype_util.is_array(lt):
                        if not isinstance(value, np.ndarray):
                            value = np.array(value)
                        if value.shape != lt.shape:
                            raise ValueError(
                                f"Cannot set array attribute '{name}' with "
                                f"value of mismatching shape:\n"
                                f"{value}"
                            )
                        for t in map(np.dtype, value.ravel()):
                            if not can_cast(t, lt.base):
                                raise TypeError(
                                    f"Cannot set attribute '{name}' "
                                    f"(C++ type {decltype(lt)}) "
                                    f"with values of {value.dtype}"
                                )
                    else:
                        rt = np.dtype(type(value))
                        if not can_cast(rt, lt):
                            raise TypeError(
                                f"Cannot set attribute '{name}' "
                                f"(C++ type {decltype(lt)}) "
                                f"with value {value} of {type(value)}"
                            )
                super().__setattr__(name, value)

            # TODO: need a nicer __repr__

        return Class

    return decor


def decltype(type, name=''):
    type = np.dtype(type, align=True)  # convert np.float32 etc. to dtype
    if _dtype_util.is_object(type):
        if len(type.names):
            return Template(r'struct{${members;};}${name}').render(
                name=name,
                members=[decltype(type.fields[v][0], v) for v in type.names])
        else:
            return f'constexpr static _empty {name} {{}}'
    elif _dtype_util.is_array(type):
        return Template(r'${type} ${name}[${shape][}]').render(
            type=decltype(type.base),
            name=name,
            shape=type.shape
        )
    else:
        if type.kind == 'S':
            return f'char {name}[{type.itemsize}]'
        else:
            return f'{str(type.name)} {name}'.strip()


def rowtype(df, pack=True, exclude=[]):
    selection = np.array([key for key in df.columns if key not in exclude])
    if pack is True:
        perm = np.argsort([-df[key].dtype.itemsize for key in selection])
        selection = selection[perm]
    packed_dtype = np.dtype([(key, df[key].dtype.newbyteorder('='))
                             for key in selection], align=True)
    return packed_dtype
