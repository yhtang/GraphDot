#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from graphdot.graph.adjacency.euclidean import Tent


class SimpleTentAtomicAdjacency:
    def __init__(self, h=1.0, order=1):
        self.adj = Tent(h * 3, order)

    def __call__(self, atom1, atom2, images, cell):
        dx = atom1.position - atom2.position
        rmin = np.linalg.norm(dx)
        for ix, iy, iz in images:
            d = dx + cell[0] * ix + cell[1] * iy + cell[2] * iz
            r = np.linalg.norm(d)
            if r < rmin:
                rmin = r
        return self.adj(np.linalg.norm(rmin)), rmin

    @property
    def cutoff(self):
        return self.adj.h * 3
