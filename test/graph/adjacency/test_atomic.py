#!/usr/bin/env python
# -*- coding: utf-8 -*-
from collections import namedtuple
import numpy as np
import pytest
from graphdot.graph.adjacency.atomic import SimpleTentAtomicAdjacency

adjacencies = [
    SimpleTentAtomicAdjacency(1.0, 1),
    SimpleTentAtomicAdjacency(1.0, 2),
    # SimpleTentAtomicAdjacency(1.0, 1, ),
]


@pytest.mark.parametrize("adj", adjacencies)
def test_atomic_adjacency(adj):
    Atom = namedtuple('Atom', ['symbol', 'position', 'index'])
    atom1 = Atom('C', np.array([0.0, 0.0, 0.0]), 0)
    atom2 = Atom('C', np.array([1.0, 0.0, 0.0]), 0)
    assert(adj.cutoff > 0)
    w, r = adj(atom1, atom2)
    assert(w >= 0)
    assert(w <= 1)
    assert(r == pytest.approx(1))
