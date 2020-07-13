#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import networkx as nx
import pytest
from graphdot.graph import Graph
from graphdot.kernel.marginalized.starting_probability import Adhoc
from graphdot.kernel.marginalized._backend_cuda import CUDABackend


def test_graph_cache():
    G = [Graph.from_networkx(nx.cycle_graph(4)) for _ in range(2)]
    backend = CUDABackend()
    p = Adhoc(lambda ns: np.ones_like(ns))
    i1, cached1, p1 = backend._register_graph(G[0], pfunc=p)
    i2, cached2, p2 = backend._register_graph(G[0], pfunc=p)
    i3, cached3, p3 = backend._register_graph(G[1], pfunc=p)
    assert(i1 == i2)
    assert(cached1 is cached2)
    assert(i1 != i3)
    assert(cached1 is not cached3)


@pytest.mark.parametrize('inplace', [True, False])
def test_graph_cache_invalidation(inplace):
    G = [Graph.from_networkx(nx.cycle_graph(4)) for _ in range(2)]
    backend = CUDABackend()
    p = Adhoc(lambda ns: np.ones_like(ns))
    cached = [backend._register_graph(g, pfunc=p) for g in G]

    if inplace:
        Graph.unify_datatype(G, inplace=inplace)
        H = G
    else:
        H = Graph.unify_datatype(G, inplace=inplace)
    for h in H:
        _, h_cached, _ = backend._register_graph(h, pfunc=p)
        for _, g_cached, _ in cached:
            assert(h_cached is not g_cached)
