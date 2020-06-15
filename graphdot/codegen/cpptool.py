#!/usr/bin/env python
# -*- coding: utf-8 -*-
import itertools as it
import numpy as np
from .typetool import can_cast, _dtype_util, decltype


def cpptype(decls=[], **kwdecls):
    """
    cpptype is a class decorator that simplifies the translation of python
    objects to corresponding C++ structures.
    """
    ctype = np.dtype(decls + list(kwdecls.items()), align=True)

    def decor(cls):
        class CppType(type):
            @property
            def dtype(self):
                """Enables numpy.dtype(cls) to return the layout of the C++
                counterpart.
                """
                return ctype

            def __repr__(self):
                return f'@cpptype({repr(ctype)})\n{repr(cls)}'

        class Class(cls, metaclass=CppType):

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
                return state

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

        Class.__name__ = cls.__name__

        return Class

    return decor
