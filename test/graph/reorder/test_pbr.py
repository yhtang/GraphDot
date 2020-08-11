#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest
import numpy as np
import networkx as nx
from graphdot.graph import Graph
from graphdot.graph.reorder import pbr


def n_tiles(A, tile_size=8):
    A = A.tocoo()
    tiles = np.unique(
        np.array([A.row // tile_size, A.col // tile_size]),
        axis=1
    )
    return len(tiles)


@pytest.mark.parametrize('n', [5, 8, 13, 20, 31, 50])
@pytest.mark.parametrize('gen', [
    lambda n: nx.wheel_graph(n),
    lambda n: nx.star_graph(n - 1),
    lambda n: nx.newman_watts_strogatz_graph(n, 3, 0.1),
    lambda n: nx.erdos_renyi_graph(n, 0.1),
])
def test_rcm_fancy_graphs(n, gen):
    g = Graph.from_networkx(gen(n))
    p = pbr(g)
    assert(np.min(p) == 0)
    assert(np.max(p) == n - 1)
    assert(len(np.unique(p)) == n)

    g_perm = g.permute(p)
    assert(n_tiles(g.adjacency_matrix) >= n_tiles(g_perm.adjacency_matrix))
