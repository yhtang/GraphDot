#!/usr/bin/env python
# -*- coding: utf-8 -*-


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
