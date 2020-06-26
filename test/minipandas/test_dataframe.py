#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pickle
import numpy as np
import pytest
from graphdot.codegen.cpptool import cpptype
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


def test_repr():
    df = DataFrame({
        'a': np.arange(3),
        'b': np.linspace(-1, 1, 3),
        'c': np.ones(3, dtype=np.bool_),
        'd': [1, 2, 3],
        'e': [(1,), (2, 3), (4, 5)],
        'f': [[1, 2, 3], [4]]
    })
    df_copy = DataFrame(eval(repr(df)))
    assert(len(df_copy) == len(df))
    assert(set(df_copy.columns) == set(df.columns))
    for key in df_copy:
        for x, y in zip(df_copy[key], df[key]):
            assert(x == y)


def test_column_insertion():
    df = DataFrame({'a': range(5)})
    assert('b' not in df)
    df['b'] = np.linspace(0, 1, 5)
    assert('b' in df)


def test_row_iteration():
    df = DataFrame({
        'a': range(5),
        'b': np.linspace(0, 1, 5)
    })
    for row in df.rows():
        assert(len(row) == 2)
        assert(row.a == row['a'])
        assert(row.b == row['b'])
        assert(row[0] == row['a'])
        assert(row[1] == row['b'])
    for i, row in df.iterrows():
        assert(isinstance(i, int))
        assert(len(row) == 2)
    for tpl in df.itertuples():
        assert(len(tpl) == 2)
    for state in df.iterstates():
        assert(len(state) == 2)


def test_column_iteration():
    df = DataFrame({
        'a': range(5),
        'b': np.linspace(0, 1, 5)
    })
    columns = [col for col in df]
    for c1, c2 in zip(columns, df.columns):
        assert(c1 == c2)


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


def test_column_attribute_access():
    df = DataFrame({
        'a': range(5),
        'b': np.linspace(0, 1, 5)
    })

    assert(df.a is not None)
    assert(df.b is not None)

    with pytest.raises(AttributeError):
        df.c


def test_simple_rowtype():
    df = DataFrame({
        'a': range(5),
        'b': np.linspace(0, 1, 5)
    })

    t1 = df.rowtype(pack=False)
    assert(len(t1.fields) == 2)
    assert(t1.names == ('a', 'b'))
    t2 = df.rowtype(pack=True)
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


@pytest.mark.parametrize('case', [
    # empty dataframe
    ([], np.dtype([], align=True)),
    # single column
    ({'x': [1, 2, 3]}, np.dtype([('x', np.int16)], align=True)),
    ({'x': [(1, 2), (2, 3)]}, np.dtype([('x', np.object)], align=True)),
    ({'x': [(1, 2), (2, 3, 4)]}, np.dtype([('x', np.object)], align=True)),
    # different types of columns
    ({'A': [1.5, 42.0],
      'B': [-1, 32768],
      'C': [False, True]},
     np.dtype([('A', np.float32),
               ('B', np.int32),
               ('C', np.bool_)], align=True)),
    # small-big-small layout optimization
    ({'A': [True, False],
      'B': [-1, 1],
      'C': [False, True]},
     np.dtype([('A', np.bool_),
               ('B', np.int16),
               ('C', np.bool_)], align=True)),
])
def test_rowtype(case):
    data, dtype = case
    df = DataFrame(data)
    assert(df.rowtype(pack=False) == dtype)
    assert(df.rowtype(pack=False).itemsize >= df.rowtype(pack=True).itemsize)


def test_pickle_empty():
    df1 = DataFrame()
    pck = pickle.dumps(df1)
    df2 = pickle.loads(pck)
    assert(len(df1) == len(df2))


def test_pickle():
    df1 = DataFrame()
    df1['x'] = np.ones(10)
    df1['y'] = np.arange(10)
    pck = pickle.dumps(df1)
    df2 = pickle.loads(pck)
    assert('x' in df2)
    assert('y' in df2)
    assert(len(df2) == len(df1))
    for x in df2.x:
        assert(x == 1)
    for i, y in enumerate(df2.y):
        assert(y == i)
