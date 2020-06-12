#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pytest
from graphdot.codegen.typetool import cpptype
from graphdot.minipandas.dataframe import DataFrame


@pytest.mark.parametrize('df', [
    DataFrame(),
    DataFrame().copy(deep=True),
    DataFrame().copy(deep=False),
])
def test_empty_df(df):
    assert(len(df) == 0)
    assert(df.columns == [])
    assert(df.rowtype() == np.dtype([]))
    assert(list(df.rows()) == [])
    assert(list(df.iterrows()) == [])
    assert(list(df.itertuples()) == [])
    assert(list(df.iterstates()) == [])
    pdf = df.to_pandas()
    assert(len(pdf) == 0)


def test_dict_init():
    df = DataFrame({'a': range(5)})
    assert('a' in df)
    assert('a' in df.columns)
    assert(len(df) == 5)
    assert(len(df.a) == 5)


def test_column_insertion():
    df = DataFrame({'a': range(5)})
    assert('b' not in df)
    df['b'] = np.linspace(0, 1, 5)
    assert('b' in df)


def test_iterators():
    df = DataFrame({
        'a': range(5),
        'b': np.linspace(0, 1, 5)
    })
    for row in df.rows():
        assert(len(row) == 2)
    for i, row in df.iterrows():
        assert(isinstance(i, int))
        assert(len(row) == 2)
    for tpl in df.itertuples():
        assert(len(tpl) == 2)
    for state in df.iterstates():
        assert(len(state) == 2)


def test_column_access():
    df = DataFrame({
        'a': range(5),
        'b': np.linspace(0, 1, 5)
    })

    for invalid_key in [1, True, 1.5, None]:
        with pytest.raises(TypeError):
            df[invalid_key]

    for nonexisting_col in ['d', 'e', 'f']:
        with pytest.raises(KeyError):
            df[nonexisting_col]

    for sel in [['a'], ['a', 'b']]:
        df_sel = df[sel]
        assert(isinstance(df_sel, DataFrame))
        assert(len(df_sel) == len(df))
        assert(len(df_sel.columns) == len(sel))

    with pytest.raises(KeyError):
        df[['x', 'y']]


def test_simple_rowtype():
    df = DataFrame({
        'a': range(5),
        'b': np.linspace(0, 1, 5)
    })

    t1 = df.rowtype(deep=True, pack=False)
    assert(len(t1.fields) == 2)
    assert(t1.names == ('a', 'b'))
    t2 = df.rowtype(deep=True, pack=True)
    assert(t2.names == ('b', 'a'))


def test_drop_column():
    df = DataFrame({
        'a': range(5),
        'b': np.linspace(0, 1, 5)
    })

    assert(list(df.drop(['a']).columns) == ['b'])
    assert(list(df.drop(['b']).columns) == ['a'])
    df_copy = df.copy()
    df_copy.drop(['a'], inplace=True)
    assert(list(df_copy.columns) == ['b'])

    with pytest.raises(KeyError):
        df.drop(['c'], inplace=True)
    assert(len(df.drop(['c']).columns) == 2)


@cpptype(x=np.int, y=np.float)
class DeepType:
    @property
    def x(self):
        return 1

    @property
    def y(self):
        return 1.5


@pytest.mark.parametrize('case', [
    ([1, 2, 3], np.int16, np.int16),
    ([(1, 2), (2, 3, 4), (3,)], np.object, np.object),
    ([DeepType(), DeepType()], DeepType.dtype, np.object)
])
def test_deep_type(case):
    data, t_deep, t_shallow = case
    df = DataFrame({'x': data})
    assert(df.rowtype(deep=True) == np.dtype([('x', t_deep)]))
    assert(df.rowtype(deep=False) == np.dtype([('x', t_shallow)]))


# rowtype_cases = [
#     # empty dataframe
#     (DataFrame([]), np.dtype([], align=True)),
#     # different types of columns
#     (DataFrame({'A': [1.5, 42.0],
#                 'B': [-1, 32768],
#                 'C': [False, True]}),
#      np.dtype([('A', np.float64),
#                ('B', np.int64),
#                ('C', np.bool_)], align=True)),
#     # small-big-small layout optimization
#     (DataFrame({'A': [True, False],
#                 'B': [-1, 1],
#                 'C': [False, True]}),
#      np.dtype([('A', np.bool_),
#                ('B', np.int64),
#                ('C', np.bool_)], align=True)),
# ]


# @pytest.mark.parametrize('case', rowtype_cases)
# def test_rowtype(case):
#     df, dtype = case
#     assert(df.rowtype(pack=False) == dtype)
#     assert(df.rowtype(pack=False).itemsize >= df.rowtype(pack=True).itemsize)
