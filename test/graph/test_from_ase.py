#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest
from ase.build import molecule
from ase.lattice.cubic import SimpleCubic
from graphdot.graph import Graph
from graphdot.graph.adjacency import AtomicAdjacency


adjacencies = [
    AtomicAdjacency(shape='tent1', length_scale=1.0, zoom=1),
    AtomicAdjacency(shape='tent2', length_scale='vdw_radius', zoom=1),
    AtomicAdjacency(
        shape='gaussian', length_scale='covalent_radius_pyykko', zoom=1.5
    ),
    AtomicAdjacency(shape='compactbell3,2'),
]


def test_ase_one():
    atoms = molecule('H2')
    graph = Graph.from_ase(atoms)
    assert(len(graph.nodes) == 2)
    assert(len(graph.edges) == 1)


@pytest.mark.parametrize('atoms', [
    SimpleCubic(latticeconstant=2, size=(2, 1, 1), symbol='Cu', pbc=(1, 0, 0)),
    SimpleCubic(latticeconstant=2, size=(1, 2, 1), symbol='Cu', pbc=(0, 1, 0)),
    SimpleCubic(latticeconstant=2, size=(1, 1, 2), symbol='Cu', pbc=(0, 0, 1)),
])
@pytest.mark.parametrize('adj', adjacencies)
def test_ase_pbc1(atoms, adj):
    graph_pbc = Graph.from_ase(atoms, use_pbc=True, adjacency=adj)
    graph_nopbc = Graph.from_ase(atoms, use_pbc=False, adjacency=adj)
    assert(len(graph_pbc.edges) == len(graph_nopbc.edges))


@pytest.mark.parametrize('atoms', [
    SimpleCubic(latticeconstant=2, size=(3, 1, 1), symbol='Cu', pbc=(1, 0, 0)),
    SimpleCubic(latticeconstant=2, size=(4, 1, 1), symbol='Cu', pbc=(1, 0, 0)),
    SimpleCubic(latticeconstant=2, size=(7, 1, 1), symbol='Cu', pbc=(1, 0, 0)),
    SimpleCubic(latticeconstant=2, size=(1, 3, 1), symbol='Cu', pbc=(0, 1, 0)),
    SimpleCubic(latticeconstant=2, size=(1, 4, 1), symbol='Cu', pbc=(0, 1, 0)),
    SimpleCubic(latticeconstant=2, size=(1, 7, 1), symbol='Cu', pbc=(0, 1, 0)),
    SimpleCubic(latticeconstant=2, size=(1, 1, 3), symbol='Cu', pbc=(0, 0, 1)),
    SimpleCubic(latticeconstant=2, size=(1, 1, 4), symbol='Cu', pbc=(0, 0, 1)),
    SimpleCubic(latticeconstant=2, size=(1, 1, 7), symbol='Cu', pbc=(0, 0, 1)),
])
def test_ase_pbc2(atoms):
    adj = AtomicAdjacency(shape='tent1', length_scale=1.0, zoom=1)
    graph_pbc = Graph.from_ase(atoms, use_pbc=True, adjacency=adj)
    graph_nopbc = Graph.from_ase(atoms, use_pbc=False, adjacency=adj)
    assert(len(graph_pbc.edges) > len(graph_nopbc.edges))


@pytest.mark.parametrize('atoms', [
    molecule('H2'),
    molecule('CH4'),
    molecule('CH3COOH'),
    SimpleCubic(latticeconstant=1, size=(3, 3, 1), symbol='Cu', pbc=(1, 1, 0)),
])
def test_ase(atoms):
    g = Graph.from_ase(atoms)
    assert(len(g.nodes) == len(atoms))
    assert(len(g.edges) > 0)
