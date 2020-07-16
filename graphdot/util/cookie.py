#!/usr/bin/env python
# -*- coding: utf-8 -*-


class VolatileCookie(dict):

    def __reduce__(self):
        return (VolatileCookie.__new__, (VolatileCookie,))

    def __deepcopy__(self, memo):
        '''Deep copy of a volatile cookie is intentionally nullified.'''
        return type(self)()
