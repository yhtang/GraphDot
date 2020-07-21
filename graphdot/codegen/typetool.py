#!/usr/bin/env python
# -*- coding: utf-8 -*-import itertools as it
import numpy as np


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
    return np.dtype(src).kind in _convertible[np.dtype(dst).kind]


class common_min_type:
    @staticmethod
    def of_values(iterable, coerce=True, min_float=np.float32,
                  ensure_signed=True):
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
        ensure_signed: bool
            Whehter to promote the result to a signed type.

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
            if ensure_signed and isinstance(r, np.dtype) and r.kind == 'u':
                r = np.promote_types(r, np.int8)
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
    def of_types(types, coerce=True, min_float=np.float32, ensure_signed=True):
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
        ensure_signed: bool
            Whehter to promote the result to a signed type.

        Returns
        -------
        t: np.dtype or class
            If `coerce=True`, t would be the smallest numpy type that can
            safely contain all the values; otherwise, t would either be the
            smallest numpy dtype or the Python class that the elements
            belongs to.
        '''
        t = None
        for r in types:
            if ensure_signed and isinstance(r, np.dtype) and r.kind == 'u':
                r = np.promote_types(r, np.int8)
            t = t or r
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
        t = None
        for i in types:
            if t is None:
                t = i
            elif t != i:
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


class _dtype_util:
    @staticmethod
    def is_object(t):
        return t.names is not None

    @staticmethod
    def is_array(t):
        return t.subdtype is not None
