#!/usr/bin/env python
# -*- coding: utf-8 -*-
import re
import copy
import functools
import numpy as np
from mendeleev import get_table
from graphdot.graph.adjacency.euclidean import Tent, Gaussian, CompactBell


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
    length[ptable.atomic_number] = ptable[name] * 0.01  # pm to A
    return length


class AtomicAdjacency:
    r"""Converts interatomic distances into edge weights using the equation
    :math:`a(i, j) = w(\frac{\lVert\mathbf{r}_{ij}\rVert}{\sigma_{ij}})`,
    where :math:`w` is a weight function that generally decays with distance,
    and :math:`\sigma_{ij}` is a length scale parameter betweens atom :math:`i`
    and :math:`j` and loosely corresponds to the typically distance of
    interatomic interactions between the atoms.

    Parameters
    ----------
    shape: str or callable
        If string, must match one of the following patterns:

        - ``tent[n]``: e.g. ``tent1``, ``teng2``, etc. :py:class:`Tent`.
        - ``gaussian``: :py:class:`Gaussian`.
        - ``compactbell[a,b]``: e.g. ``compactbell4,2``,
          :py:class:`CompactBell`.
    length_scale: str
        The atomic property to be used to determine the range and strength of
        edges to be constructed between pairs of atoms. The strength will
        generally fall to zero at roughly a distance 3 times the length scale.
        Possible values are:

        - **atomic_radius**
        - atomic_radius_rahm
        - **vdw_radius** (default)
        - vdw_radius_bondi
        - vdw_radius_truhlar
        - vdw_radius_rt
        - vdw_radius_batsanov
        - vdw_radius_dreiding
        - vdw_radius_uff
        - vdw_radius_mm3
        - vdw_radius_alvarez
        - covalent_radius_cordero
        - covalent_radius_pyykko
        - covalent_radius_bragg
        - covalent_radius_pyykko_double
        - covalent_radius_pyykko_triple
        - metallic_radius
        - metallic_radius_c12
    zoom: float
        A zooming factor to be multiplied with the length scales to extend the
        range of interactions.
    """
    def __init__(self, shape='tent1', length_scale='vdw_radius', zoom=1.0):
        if isinstance(shape, str):
            self.shape = self._parse_shape(shape)

        if isinstance(length_scale, str):
            self.ltable = get_length_scales(length_scale)
        else:
            ptbl = get_ptable()
            self.ltable = length_scale * np.ones(ptbl.atomic_number.max() + 1)

        self.ltable *= zoom

    @staticmethod
    def _parse_shape(shape):
        if shape == 'gaussian':
            return Gaussian()

        m = re.match(r'tent(\d+)', shape)
        if m:
            return Tent(ord=int(m.group(1)))

        m = re.match(r'compactbell(\d+),(\d+)', shape)
        if m:
            return CompactBell(a=int(m.group(1)), b=int(m.group(2)))

        raise ValueError(f'Unrecognizable adjacency shape: {shape}')

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
