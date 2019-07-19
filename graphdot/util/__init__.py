#!/usr/bin/env python
# -*- coding: utf-8 -*-
import warnings


def add_classmethod(cls):
    def decorate(func):
        if hasattr(cls, func.__name__):
            warnings.warn(' '.join(['Overriding', repr(cls),
                                    'existing method', repr(func)]),
                          category=RuntimeWarning)
        clsm = classmethod(func)
        clsm.__doc__ = "Add-on classmethod of %s\n\n%s" % (cls, func.__doc__)
        setattr(cls, func.__name__, clsm)
        return clsm
    return decorate
