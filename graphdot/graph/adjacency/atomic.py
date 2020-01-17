#!/usr/bin/env python
# -*- coding: utf-8 -*-
import re
import copy
import functools
import numpy as np
from mendeleev import get_table
from graphdot.graph.adjacency.euclidean import Tent


def copying_lru_cache(*args, **kwargs):

    def decorator(f):
        cached_func = functools.lru_cache(*args, **kwargs)(f)

        def wrapper(*args, **kwargs):
            return copy.deepcopy(cached_func(*args, **kwargs))
        return wrapper
    return decorator


@copying_lru_cache(maxsize=32)
def get_ptable():
    return get_table('elements')


@copying_lru_cache(maxsize=128)
def get_length_scales(name):
    ptable = get_ptable()
    length = np.zeros(ptable.atomic_number.max() + 1)
    for n, r in ptable[['atomic_number', name]].itertuples(False):
        length[n] = r * 0.01  # pm to A
    return length


class AtomicAdjacency:
    """Converts interatomic distances into edge weights

    Parameters
    ----------
    shape: str or function
        A 1D weight real-valued function that converts distance to weight.
    zoom: float
        A zooming factor to be multiplied with inferred length scales.
    images: None or list of 3D vectors
        Displacement vector of periodic images to be checked for finding
        the closest neighbors in a crystal.
    """

    def __init__(self, shape='tent1', length_scale='vdw_radius', zoom=1.0):
        if isinstance(shape, str):
            m = re.match(r'tent([1-9]\d*)', shape)
            if m:
                self.shape = Tent(ord=int(m.group(1)))
            else:
                # raise ValueError(f'Invalid shape: {shape}')
                raise ValueError('Invalid shape: {}'.format(shape))
        else:
            self.shape = shape
        if isinstance(length_scale, str):
            self.ltable = get_length_scales(length_scale)
        else:
            ptbl = get_ptable()
            self.ltable = length_scale * np.ones(ptbl.atomic_number.max() + 1)
        self.ltable *= zoom

    def __call__(self, n1, n2, r):
        """compute adjacency between atoms

        Parameters
        ----------
        n1: int
            Atomic number of the element
        n2: int
            Same as n1
        r: float
            Distance between the two atoms

        Returns
        -------
        float:
            A non-negative weight
        """
        r1 = self.ltable[n1]
        r2 = self.ltable[n2]
        weight = self.shape(r, np.sqrt(r1 * r2))
        return weight

    def cutoff(self, elements):
        max_length_scale = self.ltable[elements].max()
        return self.shape.cutoff(max_length_scale)
