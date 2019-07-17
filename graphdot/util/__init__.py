#!/usr/bin/env python
# -*- coding: utf-8 -*-


def add_classmethod(cls):
    def decorate(func):
        if hasattr(cls, func.__name__):
            raise RuntimeWarning(' '.join(['Overriding', repr(cls),
                                           'existing method', repr(func)]))
        setattr(cls, func.__name__, classmethod(func))
    return decorate
