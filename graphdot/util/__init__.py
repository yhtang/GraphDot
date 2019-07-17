#!/usr/bin/env python
# -*- coding: utf-8 -*-
import warnings


def add_classmethod(cls):
    def decorate(func):
        if hasattr(cls, func.__name__):
            warnings.warn(' '.join(['Overriding', repr(cls),
                                    'existing method', repr(func)]),
                          category=RuntimeWarning)
        setattr(cls, func.__name__, classmethod(func))
    return decorate
