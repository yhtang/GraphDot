#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest
import numpy as np
import networkx as nx
from graphdot.graph import Graph
from graphdot.graph.reorder import rcm


@pytest.mark.parametrize('n', range(2, 20))
def test_rcm_grid_graph(n):
    g = Graph.from_networkx(nx.grid_graph((n,), periodic=False))
    p = rcm(g)
    for i, j in enumerate(p[-1::-1]):
        assert(i == j)


@pytest.mark.parametrize('n', range(2, 20))
def test_rcm_complete_graph(n):
    g = Graph.from_networkx(nx.complete_graph(n))
    p = rcm(g)
    for i, j in enumerate(p[-1::-1]):
        assert(i == j)


@pytest.mark.parametrize('n', [5, 8, 13, 20, 31, 50])
@pytest.mark.parametrize('gen', [
    lambda n: nx.wheel_graph(n),
    lambda n: nx.star_graph(n - 1),
    lambda n: nx.newman_watts_strogatz_graph(n, 3, 0.1),
    lambda n: nx.erdos_renyi_graph(n, 0.2),
])
def test_rcm_fancy_graphs(n, gen):
    nxg = gen(n)
    if nxg.number_of_edges() > 0:
        g = Graph.from_networkx(nxg)
        p = rcm(g)
        assert(np.min(p) == 0)
        assert(np.max(p) == n - 1)
        assert(len(np.unique(p)) == n)
