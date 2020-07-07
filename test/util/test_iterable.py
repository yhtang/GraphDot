#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest
from graphdot.util.iterable import flatten, fold_like


def test_flatten():
    assert(tuple(flatten((1,))) == (1,))
    assert(tuple(flatten((1, 2))) == (1, 2))
    assert(tuple(flatten((1, 2, 3))) == (1, 2, 3))
    assert(tuple(flatten(((1, 2), 3))) == (1, 2, 3))
    assert(tuple(flatten(((1, 2), (3, 4)))) == (1, 2, 3, 4))
    assert(tuple(flatten(((1, (2, (3, (4,))))))) == (1, 2, 3, 4))
    assert(tuple(flatten((((((1, ), 2), 3), 4)))) == (1, 2, 3, 4))


def test_fold_like():
    assert(fold_like([1, 2, 3, 4], [0, 0, 0, 0]) == (1, 2, 3, 4))
    assert(fold_like([1, 2, 3, 4], [[0, 0], 0, 0]) == ((1, 2), 3, 4))
    assert(fold_like([1, 2, 3, 4], [0, 0, [0, 0]]) == (1, 2, (3, 4)))
    assert(fold_like([1, 2, 3, 4], [[0, 0, 0], 0]) == ((1, 2, 3), 4))
    assert(fold_like([1, 2, 3, 4], [[0, 0], [0, 0]]) == ((1, 2), (3, 4)))
    assert(fold_like([1, 2, 3, 4], [[0, 0], [0], [0]]) == ((1, 2), (3,), (4,)))
    assert(fold_like([1, 2, 3, 4], [[0]] * 4) == ((1,), (2,), (3,), (4,)))
    assert(fold_like([1, 2, 3, 4], [[[[0], 0], 0], 0]) == ((((1,), 2), 3), 4))
    assert(fold_like([1, 2, 3, 4], [0, [0, [0, [0]]]]) == (1, (2, (3, (4,)))))
