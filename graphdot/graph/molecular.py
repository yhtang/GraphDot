#!/usr/bin/env python
# -*- coding: utf-8 -*-
import itertools
import uuid
import numpy as np
import pandas as pd
from graphdot.graph import Graph
from graphdot.graph.adjacency.atomic import SimpleTentAtomicAdjacency
from graphdot.util import add_classmethod


@add_classmethod(Graph)
def from_ase(cls, atoms, use_pbc=True, adjacency='default'):

    nodes = pd.DataFrame()
    nodes['element'] = atoms.get_atomic_numbers().astype(np.int8)

    if adjacency == 'default':
        adj = SimpleTentAtomicAdjacency(h=1.0, order=1)
    else:
        adj = adjacency

    images = list(itertools.product(*tuple([-1, 0, 1] if p else [0]
                                           for p in np.logical_and(atoms.pbc,
                                                                   use_pbc))))

    edge_data = []
    for atom1 in atoms:
        for atom2 in atoms:
            if atom2.index <= atom1.index:
                continue
            w, r = adj(atom1, atom2, images, atoms.cell)
            if w > 0:
                edge_data.append(((atom1.index, atom2.index), w, r))

    edges = pd.DataFrame(edge_data, columns=['!ij', '!w', 'length'])

    return cls(nodes, edges, title='ASE Atoms {formula} {id}'.format(
               formula=atoms.get_chemical_formula(), id=uuid.uuid4().hex))
