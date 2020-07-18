#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import pickle
import pytest
from graphdot.util.cookie import VolatileCookie


def test_cookie_as_dict():
    cookie = VolatileCookie()
    cookie['a'] = 1
    assert(cookie['a'] == 1)


def test_cookie_copy():
    cookie = VolatileCookie()
    cookie[0] = True
    cookie = copy.deepcopy(cookie)
    assert(0 not in cookie)


def test_cookie_pickle():
    cookie = VolatileCookie()
    cookie[0] = 'abc'
    cookie = pickle.loads(pickle.dumps(cookie))
    assert(isinstance(cookie, VolatileCookie))
    assert(0 not in cookie)
