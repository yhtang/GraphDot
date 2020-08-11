#!/usr/bin/env python
# -*- coding: utf-8 -*-


################################################################################
# Generic hypergraph
################################################################################


class Hygr(object):

    '''
    Generic hypergraph format.
    Indexing of this class object's is 0-based.
    '''

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

        return
