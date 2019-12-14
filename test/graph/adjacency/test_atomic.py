#!/usr/bin/env python
# -*- coding: utf-8 -*-
from collections import namedtuple
import numpy as np
import pytest
from graphdot.graph.adjacency.atomic import AtomicAdjacency

adjacencies = [
    AtomicAdjacency(),
    AtomicAdjacency(shape='tent2'),
    AtomicAdjacency(length_scale=2.0),
    AtomicAdjacency(length_scale='covalent_radius_pyykko'),
    AtomicAdjacency(zoom=2.0),
]


@pytest.mark.parametrize("adj", adjacencies)
def test_atomic_adjacency(adj):
    Atom = namedtuple('Atom', ['number', 'position', 'index'])
    atom1 = Atom(6, np.array([0.0, 0.0, 0.0]), 0)
    atom2 = Atom(1, np.array([1.0, 0.0, 0.0]), 0)
    assert(adj.cutoff([atom1.number]) > 0)
    assert(adj.cutoff([atom2.number]) > 0)
    assert(adj.cutoff([atom1.number, atom2.number]) > 0)
    r = np.linalg.norm(atom1.position - atom2.position)
    w = adj(atom1.number, atom2.number, r)
    assert(w >= 0)
    assert(w <= 1)
    assert(r == pytest.approx(1))


def test_atomic_adjacency_repeated_instantiation():
    Atom = namedtuple('Atom', ['number', 'position', 'index'])
    atom1 = Atom(6, np.array([0.0, 0.0, 0.0]), 0)
    atom2 = Atom(1, np.array([1.0, 0.0, 0.0]), 0)
    r = np.linalg.norm(atom1.position - atom2.position)
    adj1 = AtomicAdjacency(zoom=0.75)
    w1 = adj1(atom1.number, atom2.number, r)
    for _ in range(10):
        adj2 = AtomicAdjacency(zoom=0.75)
        w2 = adj2(atom1.number, atom2.number, r)
        assert(w1 == w2)
