#!/usr/bin/env python
# -*- coding: utf-8 -*-
from collections import namedtuple
import numpy as np
import pandas as pd


class Series(np.ndarray):
    def __repr__(self):
        return np.array2string(self, separator=',', max_line_width=1e20)


class DataFrame:
    def __init__(self, data=None):
        self.columns = {}

        if isinstance(data, dict):
            for key, value in data.items():
                self[key] = value

    def __getitem__(self, key):
        return self.columns[key]

    def __setitem__(self, key, value):
        if not isinstance(value, np.ndarray):
            value = np.array(value).view(Series)
        self.columns[key] = value

    def __repr__(self):
        return repr(self.columns)

    def __len__(self):
        return max([0] + [len(array) for array in self.columns.values()])

    def __contains__(self, item):
        return item in self.columns

    def rows(self):
        visible = [key for key in self.columns if not key.startswith('!')]

        class RowTuple(namedtuple('RowTuple', visible)):
            def __getitem__(self, key):
                if isinstance(key, str):
                    return getattr(self, key)
                else:
                    return super().__getitem__(key)

        for i in range(len(self)):
            yield RowTuple(*[self.columns[key][i] for key in visible])

    def itertuples(self):
        yield from self.rows()

    def iterrows(self):
        yield from enumerate(self.rows())

    def to_pandas(self):
        return pd.DataFrame(self.columns)
