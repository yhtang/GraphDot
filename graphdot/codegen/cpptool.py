#!/usr/bin/env python
# -*- coding: utf-8 -*-
import itertools as it
import numpy as np
from graphdot.codegen import Template
from .typetool import can_cast, _dtype_util


def cpptype(decls=[], **kwdecls):
    """
    cpptype is a class decorator that simplifies the translation of python
    objects to corresponding C++ structures.
    """
    ctype = np.dtype(decls + list(kwdecls.items()), align=True)

    def decor(cls):
        class CppType(type(cls)):
            @property
            def dtype(self):
                '''Enables numpy.dtype(cls) to return the layout of the C++
                counterpart.'''
                return ctype

            def __repr__(self):
                return f'@cpptype({repr(ctype)})\n{repr(cls)}'

        class Class(cls, metaclass=CppType):

            @property
            def dtype(self):
                '''Useful in microkernel composition.'''
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

        Class.__name__ = cls.__name__

        return Class

    return decor


def _assert_is_identifier(name):
    if name and not name.isidentifier():
        raise ValueError(
            f'Name {name} is not a valid Python/C++ identifier.'
        )


def decltype(t, name=''):
    '''Generate C++ source code for structures corresponding to a `cpptype`
    class.'''
    t = np.dtype(t, align=True)  # convert np.float32 etc. to dtype
    if name.startswith('$'):
        ''' template class types '''
        n, t, *Ts = name[1:].split('::')
        _assert_is_identifier(n)
        return Template(r'${template}<${arguments,}>${name}').render(
            template=t,
            arguments=[decltype(T) for T in Ts],
            name=n
        )
    else:
        _assert_is_identifier(name)
        if _dtype_util.is_object(t):
            ''' structs '''
            if len(t.names):
                return Template(r'struct{${members;};}${name}').render(
                    name=name,
                    members=[decltype(t.fields[v][0], v) for v in t.names])
            else:
                return f'constexpr static _empty {name} {{}}'
        elif _dtype_util.is_array(t):
            ''' C-style arrays '''
            return Template(r'${t} ${name}[${shape][}]').render(
                t=decltype(t.base),
                name=name,
                shape=t.shape
            )
        else:
            ''' scalar types and C-strings '''
            if t.kind == 'S':
                return f'char {name}[{t.itemsize}]'
            else:
                return f'{str(t.name)} {name}'.strip()
