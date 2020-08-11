#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
from scipy.io import mmread
from scipy.io import mminfo


################################################################################
# Generic hypergraph
################################################################################


class Hygr(object):

    '''
    Generic hypergraph format.
    Indexing of this class object's is 0-based.
    '''


    def __init__ (self, unitVertexWeights=False):

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



    def _write (self, f):

        if (self._nconst == 1):
            f.write('1' + ' ' +
                    str(self._nverts) + ' ' +
                    str(self._nnets) + ' ' +
                    str(self._npins) + ' ' +
                    '3' + '\n')
        else:
            f.write('1' + ' ' +
                    str(self._nverts) + ' ' +
                    str(self._nnets) + ' ' +
                    str(self._npins) + ' ' +
                    '3' + ' ' +
                    str(self._nconst) + '\n')

        for net in xrange(self._nnets):
            f.write(str(self._nwghts[net]) + ' ')
            vstr = ' '.join([str(x+1) for x in
                             self._pins[self._xpins[net]:self._xpins[net+1]]])
            f.write(vstr + '\n')

        for v in xrange(self._nverts):
            wstr = ' '.join([str(x) for x in
                             self._cwghts[v*self._nconst:(v+1)*self._nconst]])
            f.write(wstr + '\n')

        return



    def print_hygr (self, ne=None, pe=None):

        nnets = self._nnets
        if (ne != None):
            nnets = ne
        npins = self._npins
        if (pe != None):
            npins = pe

        sys.stdout.write('#cells %d #nets %d #pins %d\n' %
                         (self._nverts, nnets, npins))
        for net in range(nnets):
            sys.stdout.write('net %4d (cost: %2d #pins: %3d): ' %
                             (net, self._nwghts[net],
                              self._xpins[net+1]-self._xpins[net]))
            for v in self._pins[self._xpins[net]:self._xpins[net+1]]:
                sys.stdout.write('%4d ' % (v))
            sys.stdout.write('\n')

        return
