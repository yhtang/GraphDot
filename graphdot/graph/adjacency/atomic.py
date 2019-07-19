#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from graphdot.graph.adjacency.euclidean import Tent


class SimpleTentAtomicAdjacency:
    def __init__(self, h=1.0, order=1, images=None):
        self.h = h
        self.adj = Tent(h * 3, order)
        self.images = images if images is not None else np.zeros((1, 3))

    def __call__(self, atom1, atom2):
        dx = atom1.position - atom2.position
        dr = np.linalg.norm(dx + self.images, axis=1)
        imin = np.argmin(dr)
        return self.adj(np.linalg.norm(dr[imin])), dr[imin]

    @property
    def cutoff(self):
        return self.h * 3
