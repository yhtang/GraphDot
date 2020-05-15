#!/usr/bin/env python
# -*- coding: utf-8 -*-
from collections import namedtuple
import numpy as np
import pandas as pd
from graphdot.codegen.typetool import common_min_type


def _as_1darray(array):
    _1darray = np.empty(len(array), dtype=common_min_type(array))
    _1darray[:] = array
    return _1darray


class Series(np.ndarray):
    '''A thin wrapper to customize serialization behavior'''
    def __repr__(self):
        return np.array2string(self, separator=',', max_line_width=1e20)


class DataFrame:
    def __init__(self, data=None):
        self._data = {}

        if isinstance(data, dict):
            for key, value in data.items():
                self[key] = value

    def __getitem__(self, key):
        if isinstance(key, str):
            return self._data[key]
        elif hasattr(key, '__iter__'):
            return self.__class__({k: self._data[k] for k in key})
        else:
            raise TypeError(f'Invalid column index {key}')

    def __setitem__(self, key, value):
        if not isinstance(value, np.ndarray):
            value = _as_1darray(value)
        self._data[key] = value.view(Series)

    def __repr__(self):
        return repr(self._data)

    def __len__(self):
        return max([0] + [len(array) for array in self._data.values()])

    def __contains__(self, item):
        return item in self._data

    def __iter__(self):
        for key in self._data:
            yield key

    @property
    def columns(self):
        return list(self._data.keys())

    def rowtype(self, pack=False):

        def element_type(key, array):
            if array.dtype != np.object:
                return array.dtype
            else:
                t = None
                for element in array:
                    t = t or element.dtype
                    if t != element.dtype:
                        raise TypeError(
                            f'Inconsistent type between rows of column {key}'
                        )
                return t

        etypes = {key: element_type(key, self[key]) for key in self.columns}
        cols = np.array(list(self.columns))
        if pack is True:
            perm = np.argsort([-etypes[key].itemsize for key in self.columns])
            cols = cols[perm]
        packed_dtype = np.dtype([(key, etypes[key].newbyteorder('='))
                                for key in cols], align=True)
        return packed_dtype

    def rows(self):
        '''Iterate over rows in the form of named tuples while skipping columns
        that do not have valid field names.'''
        visible = [key for key in self._data if key.isidentifier()]

        class RowTuple(namedtuple('RowTuple', visible)):
            def __getitem__(self, key):
                '''To support both member access and index access.'''
                if isinstance(key, str):
                    return getattr(self, key)
                else:
                    return super().__getitem__(key)

        for i in range(len(self)):
            yield RowTuple(*[self._data[key][i] for key in visible])

    def itertuples(self):
        '''Alias of `rows()` for API compatibility with pandas.'''
        yield from self.rows()

    def iterrows(self):
        '''Iterate in (row_id, row_content) tuples.'''
        yield from enumerate(self.rows())

    def iterstates(self):
        '''Iterate over rows, use the .state attribute if element is not
        scalar.'''

        for row in zip(*[self[key] for key in self.columns]):
            yield tuple(i if np.isscalar(i) else i.state for i in row)

    def to_pandas(self):
        return pd.DataFrame(self._data)

    def copy(self, deep=False):
        if deep:
            return self.__class__({
                key: np.copy(value) for key, value in self._data.items()
            })
        else:
            return self.__class__(self._data)

    def drop(self, keys, inplace=False):
        if inplace is True:
            for key in keys:
                del self._data[key]
        else:
            return self[[k for k in self.columns if k not in keys]]
