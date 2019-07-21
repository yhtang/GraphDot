#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest
from ase.build import molecule
from ase.lattice.cubic import SimpleCubic
import pymatgen
from graphdot.graph import Graph


def test_ase_one():
    atoms = molecule('H2')
    graph = Graph.from_ase(atoms)
    assert(len(graph.nodes) == 2)
    assert(len(graph.edges) == 1)


@pytest.mark.parametrize("atoms", [
    SimpleCubic(latticeconstant=2, size=(4, 4, 2), symbol='Cu', pbc=(1, 1, 0)),
    SimpleCubic(latticeconstant=2, size=(3, 3, 1), symbol='Cu', pbc=(1, 1, 0)),
])
def test_ase_pbc(atoms):
    graph_pbc = Graph.from_ase(atoms, use_pbc=True)
    graph_nopbc = Graph.from_ase(atoms, use_pbc=False)
    print(graph_pbc.edges)
    print(graph_nopbc.edges)
    assert(len(graph_pbc.edges) > len(graph_nopbc.edges))
    graph_nopbcx = Graph.from_ase(atoms, use_pbc=[False, True, True])
    graph_nopbcy = Graph.from_ase(atoms, use_pbc=[True, False, True])
    graph_nopbcz = Graph.from_ase(atoms, use_pbc=[True, True, False])
    assert(len(graph_pbc.edges) > len(graph_nopbcx.edges))
    assert(len(graph_pbc.edges) > len(graph_nopbcy.edges))
    assert(len(graph_pbc.edges) == len(graph_nopbcz.edges))
    assert(len(graph_nopbcx.edges) == len(graph_nopbcy.edges))


@pytest.mark.parametrize("atoms", [
    molecule('H2'),
    molecule('CH4'),
    molecule('CH3COOH'),
    SimpleCubic(latticeconstant=1, size=(3, 3, 1), symbol='Cu', pbc=(1, 1, 0)),
])
def test_ase(atoms):
    g = Graph.from_ase(atoms)
    assert(len(g.nodes) == len(atoms))
    assert(len(g.edges) > 0)
