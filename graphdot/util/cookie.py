#!/usr/bin/env python
# -*- coding: utf-8 -*-


class VolatileCookie(dict):

    def __getstate__(self):
        return None

    def __setstate__(self, state):
        pass

    def __deepcopy__(self, memo):
        '''Deep copy of a volatile cookie is intentionally nullified.'''
        return type(self)()
