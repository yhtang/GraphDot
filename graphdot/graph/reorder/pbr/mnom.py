#!/usr/bin/env python
# -*- coding: utf-8 -*-
import math
import kahypar as kahypar
import numpy as np
from .config import to_ini
from .colnet_hypergraph import ColnetHygr
from .hypergraph import Hygr


class PbrMnom:
    '''Partitioning-based reordering.

    Parameters
    ----------
    tilesize: int, default=8
        Size of the tile to be used for partitioning
    mnc: int, default=100
        Message net cost. The higher the value,  the more aggressive the
        will try to minimize the number of nonempty tiles.
    addMsgNets: bool, default=True
        Whether to add message nets to minimize the number of nonempty tiles.
        Should be True in most cases.
    '''

    def __init__(self, tilesize=8, mnc=100, addMsgNets=True, config=None):

        self._tilesize = tilesize
        self._mnc = mnc
        self._addMsgNets = addMsgNets
        self._context = kahypar.Context()
        with to_ini(config) as ini:
            self._context.loadINIconfiguration(ini)

    def _bisect(self, curlvl, lpvec, nextlvl, idx):

        lidx = 2*idx
        ridx = 2*idx + 1
        h = curlvl[idx][0]
        map = [None] * h._nverts
        hl = Hygr()
        hl._nverts = hl._nnets = hl._npins = 0
        hr = Hygr()
        hr._nverts = hr._nnets = hr._npins = 0

        # assign vertices
        for v in range(h._nverts):
            if (lpvec[v] == 0):
                map[v] = hl._nverts
                hl._nverts += 1
            else:
                map[v] = hr._nverts
                hr._nverts += 1

        # unit vertex weights
        hl._cwghts = [1] * hl._nverts
        nextlvl[lidx][2] = [None] * hl._nverts
        hr._cwghts = [1] * hr._nverts
        nextlvl[ridx][2] = [None] * hr._nverts

        # global vertex ids
        for v in range(h._nverts):
            if (lpvec[v] == 0):
                nextlvl[lidx][2][map[v]] = curlvl[idx][2][v]
            else:
                nextlvl[ridx][2][map[v]] = curlvl[idx][2][v]

        # split nets and pins
        lpin = lnet = rpin = rnet = 0
        for n in range(h._nnets):
            ls = lpin
            rs = rpin
            for v in h._pins[h._xpins[n]:h._xpins[n+1]]:
                if (lpvec[v] == 0):
                    lpin += 1
                else:
                    rpin += 1
            if (lpin > ls):
                lnet += 1
            if (rpin > rs):
                rnet += 1

        hl._xpins = [0] * (lnet + 2)
        hl._pins = [0] * lpin
        # hl._nwghts = [0] * (lnet + 2)
        hl._nwghts = [0] * lnet
        hr._xpins = [0] * (rnet + 2)
        hr._pins = [0] * rpin
        # hr._nwghts = [0] * (rnet + 2)
        hr._nwghts = [0] * rnet

        for n in range(h._nnets):
            for v in h._pins[h._xpins[n]:h._xpins[n+1]]:
                if (lpvec[v] == 0):
                    hl._pins[hl._npins] = map[v]
                    hl._npins += 1
                else:
                    hr._pins[hr._npins] = map[v]
                    hr._npins += 1

            if (hl._xpins[hl._nnets] < hl._npins):
                hl._xpins[hl._nnets + 1] = hl._npins
                hl._nwghts[hl._nnets] = h._nwghts[n]
                hl._nnets += 1

            if (hr._xpins[hr._nnets] < hr._npins):
                hr._xpins[hr._nnets + 1] = hr._npins
                hr._nwghts[hr._nnets] = h._nwghts[n]
                hr._nnets += 1

        nextlvl[lidx][0] = hl
        nextlvl[ridx][0] = hr

        # no super-nets for now
        nextlvl[lidx][3] = hl._nnets
        nextlvl[lidx][4] = hl._npins
        nextlvl[ridx][3] = hr._nnets
        nextlvl[ridx][4] = hr._npins

    def _add_send_msg_nets(self, horig, curpid, cur_nhygrs,
                           gpvec, curlvl, idx):

        lb = curpid
        ub = curpid + 1
        nsendnets = 0
        send_net_ids = [0] * cur_nhygrs
        send_net_szs = [0] * cur_nhygrs
        sends = [0] * cur_nhygrs
        totalpin = 0
        hcur = curlvl[idx][0]

        for v in range(hcur._nverts):
            vorig = curlvl[idx][2][v]  # also a net
            for u in horig._pins[horig._xpins[vorig]:horig._xpins[vorig+1]]:
                p = gpvec[u]
                if ((p < lb or p >= ub) and sends[p] == 0):
                    if (send_net_szs[p] == 0):
                        send_net_ids[p] = nsendnets
                        nsendnets += 1
                    send_net_szs[p] += 1
                    sends[p] = 1
                    totalpin += 1

            for i in range(cur_nhygrs):
                sends[i] = 0

        ne = curlvl[idx][3]
        pe = curlvl[idx][4]

        # print(len(hcur._nwghts), ne)
        assert (len(hcur._xpins) == (ne+2) and
                len(hcur._pins) == pe and
                len(hcur._nwghts) == ne)
        hcur._xpins.extend([0] * nsendnets)
        hcur._pins.extend([0] * totalpin)
        hcur._nwghts.extend([0] * nsendnets)

        for n in range(cur_nhygrs):
            if (send_net_szs[n] != 0):
                hcur._xpins[ne+2+send_net_ids[n]] = send_net_szs[n]

        for n in range(ne+1, ne+nsendnets+2):
            hcur._xpins[n] += hcur._xpins[n-1]

        for v in range(hcur._nverts):
            vorig = curlvl[idx][2][v]  # also a net
            for u in horig._pins[horig._xpins[vorig]:horig._xpins[vorig+1]]:
                p = gpvec[u]
                if ((p < lb or p >= ub) and sends[p] == 0):
                    hcur._pins[hcur._xpins[ne+1+send_net_ids[p]]] = v
                    hcur._xpins[ne+1+send_net_ids[p]] += 1
                    sends[p] = 1

            for i in range(cur_nhygrs):
                sends[i] = 0

        ncost = 2 * self._mnc * 10
        for n in range(ne, ne+nsendnets):
            hcur._nwghts[n] = ncost
        for n in range(hcur._nnets):
            hcur._nwghts[n] = 10

        curlvl[idx][3] = ne + nsendnets
        curlvl[idx][4] = hcur._xpins[curlvl[idx][3]]
        hcur._xpins[curlvl[idx][3]+1] = 0

    def partition_hygr(self, h):

        if (h._nverts <= self._tilesize):
            return range(h._nverts)

        tilesize = self._tilesize
        nparts = int((h._nverts + tilesize - 1) / tilesize)
        curnhygr = 1
        kway_bound = nparts
        totalcut = 0
        gpvec = [0] * h._nverts

        context = self._context
        context.setK(2)         # will always partition into 2

        # nparts should be > 1 at this point
        assert(nparts > 1)
        if (nparts & (nparts-1) != 0):
            kway_bound = int(math.pow(2, int(math.log2(nparts)) + 1))

        # each entry three elems: [hygr, curk, gcids, ne, pe]
        # ne and pe include supernets
        curlvl = [None]
        nextlvl = [[None] * 5, [None] * 5]

        curlvl[0] = [h, nparts, [x for x in range(h._nverts)],
                     h._nnets, h._npins]

        while (curnhygr != kway_bound):

            lastid = 0

            for i in range(curnhygr):

                if (curlvl[i][1] == 1):
                    lastid += 1
                    continue

                if (curlvl[i][0] is None):
                    continue

                if (curnhygr > 1 and self._addMsgNets):
                    self._add_send_msg_nets(h, lastid, 2*curnhygr,
                                            gpvec, curlvl, i)

                # compute target pw
                curk = curlvl[i][1]
                tpw = [None] * 2
                if (curlvl[i][0]._nverts % tilesize != 0):
                    tmp = int((curk + 1) / 2)
                    tpw[0] = tmp * tilesize
                    tpw[1] = curlvl[i][0]._nverts - tpw[0]
                else:
                    if (curk % 2 == 0):
                        tpw[0] = tpw[1] = (curk / 2) * tilesize
                    else:
                        tmp = int(curk / 2)
                        tpw[0] = (tmp + 1) * tilesize
                        tpw[1] = tmp * tilesize

                tpw[0] = int(tpw[0])
                tpw[1] = int(tpw[1])

                # form kahypar hypergraph
                hk = kahypar.Hypergraph(
                    curlvl[i][0]._nverts, curlvl[i][3],  # with msg nets
                    curlvl[i][0]._xpins, curlvl[i][0]._pins,
                    2, curlvl[i][0]._nwghts, curlvl[i][0]._cwghts)
                context.setCustomTargetBlockWeights(tpw)
                context.suppressOutput(True)

                # partition and get part vector
                kahypar.partition(hk, context)
                lpvec = [None] * curlvl[i][0]._nverts
                for v in range(curlvl[i][0]._nverts):
                    lpvec[v] = hk.blockID(v)

                # @TODO if tpw is not achieved, return id perm
                curcut = kahypar.cut(hk)
                totalcut += curcut
                assert(hk.blockWeight(0) == tpw[0] and
                       hk.blockWeight(1) == tpw[1])

                # update global partvec
                for v in range(h._nverts):  # orig hygr
                    if (gpvec[v] > lastid):
                        gpvec[v] += 1
                for v in range(curlvl[i][0]._nverts):
                    if (lpvec[v] == 1):
                        gpvec[curlvl[i][2][v]] += 1

                lastid += 2
                left = 2*i
                right = 2*i + 1
                nextlvl[right][1] = int(curk / 2)
                nextlvl[left][1] = nextlvl[right][1] + (curk % 2)

                # bisect
                self._bisect(curlvl, lpvec, nextlvl, i)

            curnhygr = curnhygr * 2

            # update curlvl and nextlvl
            if (curnhygr != nparts):
                curlvl = nextlvl
                nextlvl = [[None] * 5 for i in range(2*curnhygr)]

        return gpvec

    def __call__(self, row_ids, col_ids, nrow, ncol):
        '''Reorder a graph (as represented by a symmetric sparse matrix) using
        PBR and return the permutation array.

        Parameters
        ----------
        row_ids: sequence
            Row indices of the non-zero elements.
        col_ids: sequence
            Column indices of the non-zero elements.
        nrow: int
            Number of rows
        ncol: int
            Number of columns

        Returns
        -------
        perm: ndarray
            Array of permuted row/column indices.
        '''
        h = ColnetHygr(True)
        h.createFromPairs(row_ids, col_ids, nrow, ncol)
        pvec = self.partition_hygr(h)
        perm = [(v, pvec[v]) for v in range(nrow)]
        perm = sorted(perm, key=lambda x: x[1])
        perm = [x[0] for x in perm]

        return np.array(perm)
