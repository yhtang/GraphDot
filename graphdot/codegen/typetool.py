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


class common_min_type:
    @staticmethod
    def of_values(iterable, coerce=True, min_float=np.float32):
        '''Find the common minimum elemet type that can safely hold all
        elements of an iterable sequence.

        Parameters
        ----------
        iterable:
            Sequence of objects for which a common type is to be inferred.
        coerce: bool
            Whether or not to up-cast in case when elements have different
            but inter-converible types.
        min_float: dtype
            The smallest floating-point type that should be returned.
            Introduced because float16 is not universally supported yet.

        Returns
        -------
        t: np.dtype or class
            If `coerce=True`, t would be the smallest numpy type that can
            safely contain all the values; otherwise, t would either be the
            smallest numpy dtype or the Python class that the elements
            belongs to.
        '''
        t = None
        for i in iterable:
            r = np.min_scalar_type(i) if np.isscalar(i) else type(i)
            t = t or r
            if t != r:
                if coerce:
                    t = np.promote_types(t, r)
                else:
                    return None

        if isinstance(t, np.dtype) and t.kind == 'f':
            t = np.promote_types(t, min_float)

        return t

    @staticmethod
    def of_types(types, coerce=True, min_float=np.float32):
        '''Find the common minimum elemet type that can safely hold all types
        in the given list.

        Parameters
        ----------
        iterable:
            Sequence of types for which a common type is to be inferred.
        coerce: bool
            Whether or not to up-cast in case when elements have different
            but inter-converible types.
        min_float: dtype
            The smallest floating-point type that should be returned.
            Introduced because float16 is not universally supported yet.

        Returns
        -------
        t: np.dtype or class
            If `coerce=True`, t would be the smallest numpy type that can
            safely contain all the values; otherwise, t would either be the
            smallest numpy dtype or the Python class that the elements
            belongs to.
        '''
        t = next(iter(types))
        for r in types:
            if t != r:
                if coerce:
                    t = np.promote_types(t, r)
                else:
                    return None

        if isinstance(t, np.dtype) and t.kind == 'f':
            t = np.promote_types(t, min_float)

        return t


class common_concrete_type:
    @staticmethod
    def of_values(iterable):
        '''Find the common concrete type (if one exists) of all elements in a
        sequence.

        Returns
        -------
        t: type or None
            Returns the type of the elements if all elements share a common
            type, and returns `None` otherwise.
        '''
        return common_concrete_type.of_types(map(type, iterable))

    @staticmethod
    def of_types(types):
        '''Find the common concrete type (if one exists) among a type list.

        Returns
        -------
        t: type or None
            Returns the type of the elements if all elements share a common
            type, and returns `None` otherwise.
        '''
        t = next(iter(types))
        for i in types:
            if t != i:
                return None
        return t


def have_same_fields(t1, t2):
    if bool(t1.fields) != bool(t2.fields):
        return False
    if t1.fields and t2.fields:
        if len(t1.fields) != len(t2.fields):
            return False
        if set(t1.fields) != set(t2.fields):
            return False
        for f in t1.fields:
            if not have_same_fields(t1.fields[f][0], t2.fields[f][0]):
                return False
    return True


class Mangler:
    @staticmethod
    def mangle(identifier, ):
        pass
        # tag = '${key}::frozen_array::{dtype}'.format(
        #     key=key,
        #     dtype=inner_type.str
        # )

    @staticmethod
    def demangle():
        pass


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

        Class.__name__ = clss.__name__

        return Class

    return decor


def _assert_is_identifier(name):
    if name and not name.isidentifier():
        raise ValueError(
            f'Name {name} is not a valid Python/C++ identifier.'
        )


def decltype(t, name=''):
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
