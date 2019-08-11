#!/usr/bin/env python
# -*- coding: utf-8 -*-
import re
import numpy as np
from mendeleev import Element
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


class AtomicAdjacency:
    """Converts interatomic distances into edge weights

    Parameters
    ----------
    zoom: float
        A zooming factor to be multiplied with inferred length scales.
    shape: str or function
        A 1D weight real-valued function that converts distance to weight.
    images: None or list of 3D vectors
        Displacement vector of periodic images to be checked for finding
        the closest neighbors in a crystal.
    """

    def __init__(self, zoom=1, shape='tent1', images=None):
        self.zoom = zoom
        self.images = images
        if isinstance(shape, str):
            m = re.match('tent([1-9]\d*)', shape)
            if m:
                self.shape= Tent(ord = int(m.group(1)))
            else:
                raise ValueError('Invalid shape: {}'.format(shape))
        else:
            self.shape = shape


    def __call__(self, atom1, atom2):
        """compute adjacency between atoms

        Parameters
        ----------
        atom1: :py:class:`ase.Atom` instance
            An atom object with properties such as `position`, `number`, `index`, etc.
        atom2: :py:class:`ase.Atom` instance
            Same as `atom1`

        Returns
        -------
        (float, float):
            A non-negative weight and a min distance
        """
        dx = atom1.position - atom2.position
        dr = np.linalg.norm(dx + self.images, axis=1)
        imin = np.argmin(dr)
        r1 = Element(atom1.number).covalent_radius
        r2 = Element(atom2.number).covalent_radius
        weight = self.shape(
            dr[imin],
            self.zoom * np.sqrt(r1 * r2)
        )
        return weight, dr[imin]
