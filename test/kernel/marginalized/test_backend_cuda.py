#!/usr/bin/env python
# -*- coding: utf-8 -*-
import networkx as nx
import pytest
from graphdot.graph import Graph
from graphdot.kernel.marginalized._backend_cuda import CUDABackend


def test_graph_cache():
    G = [Graph.from_networkx(nx.cycle_graph(4)) for _ in range(2)]
    backend = CUDABackend()
    cached1 = backend._register_graph(G[0], p=lambda _1, _2: 1)
    cached2 = backend._register_graph(G[0], p=lambda _1, _2: 1)
    cached3 = backend._register_graph(G[1], p=lambda _1, _2: 1)
    assert(cached1 is cached2)
    assert(cached1 is not cached3)


@pytest.mark.parametrize('inplace', [True, False])
def test_graph_cache_invalidation(inplace):
    G = [Graph.from_networkx(nx.cycle_graph(4)) for _ in range(2)]
    backend = CUDABackend()
    cached = [backend._register_graph(g, p=lambda _1, _2: 1) for g in G]

    if inplace:
        Graph.unify_datatype(G, inplace=inplace)
        H = G
    else:
        H = Graph.unify_datatype(G, inplace=inplace)
    for h in H:
        h_cached = backend._register_graph(h, p=lambda _1, _2: 1)
        for g_cached in cached:
            assert(h_cached is not g_cached)
