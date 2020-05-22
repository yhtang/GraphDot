#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pytest
from graphdot.minipandas import DataFrame


rowtype_cases = [
    # empty dataframe
    (DataFrame([]), np.dtype([], align=True)),
    # different types of columns
    (DataFrame({'A': [1.5, 42.0],
                'B': [-1, 32768],
                'C': [False, True]}),
     np.dtype([('A', np.float64),
               ('B', np.int64),
               ('C', np.bool_)], align=True)),
    # small-big-small layout optimization
    (DataFrame({'A': [True, False],
                'B': [-1, 1],
                'C': [False, True]}),
     np.dtype([('A', np.bool_),
               ('B', np.int64),
               ('C', np.bool_)], align=True)),
]


@pytest.mark.parametrize('case', rowtype_cases)
def test_rowtype(case):
    df, dtype = case
    # assert(rowtype(df, pack=False) == dtype)
    assert(df.rowtype(pack=False).itemsize >= df.rowtype(pack=True).itemsize)
