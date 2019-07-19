#!/usr/bin/env python
# -*- coding: utf-8 -*-
from itertools import product
import uuid
import numpy as np
import pandas as pd
import ase
import pymatgen
import pymatgen.io.ase
from graphdot.graph import Graph
from graphdot.graph.adjacency.atomic import SimpleTentAtomicAdjacency
from graphdot.util import add_classmethod


@add_classmethod(Graph)
def from_molecule(cls, molecule, use_pbc=True, adjacency='default'):
    """Convert molecules to graphs

    Parameters
    ----------
    atoms: an ASE Atoms or pymatgen Molecule object
        A molecule as represented by a collection of atoms in 3D space.
    usb_pbc: boolean or list of 3 booleans
        Whether to use the periodic boundary condition as specified in the
        atoms object to create edges between atoms.
    adjacency: 'default' or object
        A functor that implements the rule for creating edges between atoms.

    Returns
    -------
    Graph:
        a molecular graph where atoms become nodes while edges resemble short-
        range interatomic interactions.
    """
    if isinstance(molecule, ase.Atoms):
        atoms = molecule
    elif isinstance(molecule, pymatgen.Molecule):
        atoms = pymatgen.io.ase.AseAtomsAdaptor.get_atoms(molecule)
    else:
        raise TypeError('Unknown molecule format %s' % type(molecule))

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

    return cls(nodes, edges, title='Molecule {formula} {id}'.format(
               formula=atoms.get_chemical_formula(), id=uuid.uuid4().hex))
