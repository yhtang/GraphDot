#!/usr/bin/env python
# -*- coding: utf-8 -*-
import time
from collections import OrderedDict


def add_classmethod(cls, override=False):
    def decorate(func):
        if override is not True and hasattr(cls, func.__name__):
            raise RuntimeError('Class %s already has a method named %s' % (
                repr(cls), func.__name__))
        clsm = classmethod(func)
        clsm.__doc__ = "Add-on classmethod of %s\n\n%s" % (cls, func.__doc__)
        setattr(cls, func.__name__, clsm)
        return clsm
    return decorate


class Timer:
    def __init__(self):
        self.reset()

    def reset(self):
        self.t = OrderedDict()
        self.dt = OrderedDict()

    def tic(self, tag):
        self.t[tag] = time.perf_counter()

    def toc(self, tag):
        self.dt[tag] = time.perf_counter() - self.t[tag]
        del self.t[tag]

    def report(self, unit='s'):
        if unit == 's':
            scale = 1.0
        elif unit == 'ms':
            scale = 1e3
        elif unit == 'us':
            scale = 1e6
        elif unit == 'ns':
            scale = 1e9
        else:
            raise ValueError('Unknown unit %s' % unit)
        for tag, dt in self.dt.items():
            print('%9.1f %s on %s' % (dt * scale, unit, tag))
