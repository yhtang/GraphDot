#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Adaptor for PyMatgen's Molecule objects"""
import pymatgen.io


def _from_pymatgen(cls, molecule, use_pbc=True, adjacency='default'):
    """Convert from pymatgen molecule to molecular graph

    Parameters
    ----------
    molecule: pymatgen Molecule object
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
    atoms = pymatgen.io.ase.AseAtomsAdaptor.get_atoms(molecule)
    return cls.from_ase(atoms, use_pbc, adjacency)
