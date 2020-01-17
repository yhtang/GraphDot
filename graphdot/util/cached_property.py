#!/usr/bin/env python
# -*- coding: utf-8 -*-
from _thread import RLock

_NOT_FOUND = object()


# TODO: to be removed after Python 3.8
class cached_property:
    def __init__(self, func):
        self.func = func
        self.attrname = None
        self.__doc__ = func.__doc__
        self.lock = RLock()

    def __set_name__(self, owner, name):
        if self.attrname is None:
            self.attrname = name
        elif name != self.attrname:
            raise TypeError(
                "Cannot assign the same cached_property "
                "to two different names "
                # f"({self.attrname!r} and {name!r})."
                "({!r} and {name!r}).".format(self.attrname, name)
            )

    def __get__(self, instance, owner=None):
        if instance is None:
            return self
        if self.attrname is None:
            raise TypeError(
                "Cannot use cached_property instance without"
                "calling __set_name__ on it.")
        try:
            cache = instance.__dict__
        except AttributeError:
            msg = (
                # f"No '__dict__' attribute on {type(instance).__name__!r} "
                # f"instance to cache {self.attrname!r} property."
                "No '__dict__' attribute on {!r} "
                "instance to cache {!r} property.".format(
                    type(instance).__name__,
                    self.attrname
                )
            )
            raise TypeError(msg) from None
        val = cache.get(self.attrname, _NOT_FOUND)
        if val is _NOT_FOUND:
            with self.lock:
                # check if another thread filled cache while we awaited lock
                val = cache.get(self.attrname, _NOT_FOUND)
                if val is _NOT_FOUND:
                    val = self.func(instance)
                    try:
                        cache[self.attrname] = val
                    except TypeError:
                        msg = (
                            "The '__dict__' attribute on "
                            # f"{type(instance).__name__!r} instance "
                            "{!r} instance "
                            "does not support item assignment for "
                            # f"caching {self.attrname!r} property."
                            "caching {!r} property.".format(
                                type(instance).__name__,
                                self.attrname,
                            )
                        )
                        raise TypeError(msg) from None
        return val
