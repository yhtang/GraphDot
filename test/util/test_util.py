#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest
from graphdot.util import add_classmethod


def test_add_classmethod():
    class Class:
        pass

    assert(not hasattr(Class, 'fun'))

    @add_classmethod(Class)
    def fun(cls):
        return True

    assert(hasattr(Class, 'fun'))
    assert(Class.fun())

    with pytest.warns(RuntimeWarning):
        @add_classmethod(Class)
        def fun(cls):
            return False

    assert(hasattr(Class, 'fun'))
    assert(not Class.fun())
