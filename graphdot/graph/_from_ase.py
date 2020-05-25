#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Adaptor for ASE's Atom objects"""
from itertools import product
import uuid
import numpy as np
from scipy.spatial import cKDTree
from graphdot.graph.adjacency.atomic import AtomicAdjacency
from graphdot.minipandas import DataFrame


def _from_ase(cls, atoms, use_charge=False, use_pbc=True,
              adjacency='default'):
    """Convert from ASE atoms to molecular graph

    Parameters
    ----------
    atoms: ASE Atoms object
        A molecule as represented by a collection of atoms in 3D space.
    usb_pbc: boolean or list of 3 booleans
        Whether to use the periodic boundary condition as specified in the
        atoms object to create edges between atoms.
    adjacency: 'default' or object
        A functor that implements the rule for making edges between atoms.

    Returns
    -------
    graphdot.Graph:
        a molecular graph where atoms become nodes while edges resemble
        short-range interatomic interactions.
    """
    if adjacency == 'default':
        adjacency = AtomicAdjacency()

    nodes = DataFrame({'!i': range(len(atoms))})
    nodes['element'] = atoms.get_atomic_numbers().astype(np.int8)
    if use_charge:
        nodes['charge'] = atoms.get_initial_charges().astype(np.float32)

    pbc = np.logical_and(atoms.pbc, use_pbc)
    images = [(atoms.cell.T * image).sum(axis=1) for image in product(
        *tuple([-1, 0, 1] if p else [0] for p in pbc))]
    x = atoms.get_positions()
    x_images = np.vstack([x + i for i in images])
    # prefer lookup over integer modulo
    j_images = list(range(len(atoms))) * len(images)

    cutoff = adjacency.cutoff(atoms.get_atomic_numbers())
    nl = cKDTree(x).sparse_distance_matrix(cKDTree(x_images), cutoff)

    edgedict = {}
    for (i, j), r in nl.items():
        j = j_images[j]
        if j > i:
            w = adjacency(atoms[i].number, atoms[j].number, r)
            if w > 0 and ((i, j) not in edgedict or edgedict[(i, j)][1] > r):
                edgedict[(i, j)] = (w, r)
    i, j, w, r = list(zip(*[(i, j, w, r)
                            for (i, j), (w, r) in edgedict.items()]))

    edges = DataFrame({
        '!i': np.array(i, dtype=np.uint32),
        '!j': np.array(j, dtype=np.uint32),
        '!w': np.array(w, dtype=np.float32),
        'length': np.array(r, dtype=np.float32),
    })

    return cls(nodes, edges, title='Molecule {formula} {id}'.format(
                formula=atoms.get_chemical_formula(), id=uuid.uuid4().hex))
