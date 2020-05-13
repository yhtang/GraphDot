#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Adaptor for RDKit's Molecule objects"""
import networkx as nx
from ._from_networkx import _from_networkx


def _from_rdkit(cls, mol):
    g = nx.Graph()

    for i, atom in enumerate(mol.GetAtoms()):
        g.add_node(i)
        g.nodes[i]['symbol'] = atom.GetAtomicNum()
        g.nodes[i]['charge'] = atom.GetFormalCharge()
        g.nodes[i]['hcount'] = atom.GetTotalNumHs()
        g.nodes[i]['hybridization'] = atom.GetHybridization()
        g.nodes[i]['aromatic'] = atom.GetIsAromatic()
        g.nodes[i]['chiral'] = atom.GetChiralTag()

    for bond in mol.GetBonds():
        ij = (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
        g.add_edge(*ij)
        g.edges[ij]['bondtype'] = bond.GetBondType()
        g.edges[ij]['aromatic'] = bond.GetIsAromatic()
        g.edges[ij]['conjugated'] = bond.GetIsConjugated()
        g.edges[ij]['stereo'] = bond.GetStereo()
        g.edges[ij]['inring'] = bond.IsInRing()

    return _from_networkx(cls, g)
