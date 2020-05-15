#!/usr/bin/env python
# -*- coding: utf-8 -*-
import itertools as it
import numpy as np
from graphdot.codegen import Template


__all__ = ['cpptype', 'decltype']


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


def common_min_type(iterable, coerce=True, min_float=np.float32):
    '''Find the common minimum scalar type that can safely hold all elements of
    an iterable.'''
    t = None
    for i in iterable:
        r = np.min_scalar_type(i) if np.isscalar(i) else np.object
        t = t or r
        if t != r and not coerce:
            return None
        t = np.promote_types(t, r)

    if coerce and t.kind == 'f':
        t = np.promote_types(t, min_float)

    return t


class _dtype_util:
    @staticmethod
    def is_object(t):
        return t.names is not None

    @staticmethod
    def is_opaque_object(t):
        return t.kind == 'O'

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


def decltype(t, name='', custom_types={}):
    if name and not name.isidentifier():
        raise ValueError(
            f'Name {name} is not a valid Python/C++ identifier.'
        )
    t = np.dtype(t, align=True)  # convert np.float32 etc. to dtype
    if _dtype_util.is_opaque_object(t):
        if t not in custom_types:
            raise TypeError(
                f'No custom callback provided for opaque type {t.name}'
            )
        return custom_types[t.name](t)  # WIP
    elif _dtype_util.is_object(t):
        if len(t.names):
            return Template(r'struct{${members;};}${name}').render(
                name=name,
                members=[decltype(t.fields[v][0], v) for v in t.names])
        else:
            return f'constexpr static _empty {name} {{}}'
    elif _dtype_util.is_array(t):
        return Template(r'${t} ${name}[${shape][}]').render(
            t=decltype(t.base),
            name=name,
            shape=t.shape
        )
    else:
        if t.kind == 'S':
            return f'char {name}[{t.itemsize}]'
        else:
            return f'{str(t.name)} {name}'.strip()
