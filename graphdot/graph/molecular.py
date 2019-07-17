#!/usr/bin/env python
# -*- coding: utf-8 -*-
from itertools import product
import uuid
import numpy as np
import pandas as pd
from graphdot.graph import Graph
from graphdot.graph.adjacency.atomic import SimpleTentAtomicAdjacency
from graphdot.util import add_classmethod


@add_classmethod(Graph)
def from_ase(cls, atoms, use_pbc=True, adjacency='default'):

    pbc = np.logical_and(atoms.pbc, use_pbc)
    images = [(atoms.cell.T * image).sum(axis=1) for image in product(
        *tuple([-1, 0, 1] if p else [0] for p in pbc))]

    if adjacency == 'default':
        adj = SimpleTentAtomicAdjacency(h=1.0, order=1, images=images)
    else:
        adj = adjacency

    nodes = pd.DataFrame()
    nodes['element'] = atoms.get_atomic_numbers().astype(np.int8)

    edge_data = []
    for atom1 in atoms:
        for atom2 in atoms:
            if atom2.index <= atom1.index:
                continue
            w, r = adj(atom1, atom2)
            if w > 0:
                edge_data.append(((atom1.index, atom2.index), w, r))

    edges = pd.DataFrame(edge_data, columns=['!ij', '!w', 'length'])

    return cls(nodes, edges, title='ASE Atoms {formula} {id}'.format(
               formula=atoms.get_chemical_formula(), id=uuid.uuid4().hex))
