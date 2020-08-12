#!/usr/bin/env python
# -*- coding: utf-8 -*-


class Hygr(object):
    '''A generic hypergraph format. Uses 0-based indexing.'''

    def __init__(self, unitVertexWeights=False):
        self._nverts = None
        self._nnets = None
        self._npins = None
        self._pins = None
        self._xpins = None
        self._nwghts = None
        self._cwghts = None
        self._nconst = None
        self._unitVertexWeights = unitVertexWeights
