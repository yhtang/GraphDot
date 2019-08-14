import pytest
import numpy as np
import networkx as nx
from graphdot.graph import Graph


@pytest.mark.parametrize("n", [10, 100, 1000])
def test_graph_from_networkx(n, benchmark):

    nxg = nx.Graph(title='Large')
    nn = n
    ne = n * 10
    np.random.seed(0)
    for i in range(nn):
        nxg.add_node(i, label=np.random.randn())
    for _ in range(ne):
        i, j = np.random.randint(nn, size=2)
        nxg.add_edge(i, j, weight=np.random.rand())

    def fun(nxg):
        return Graph.from_networkx(nxg)

    g = benchmark.pedantic(fun, args=(nxg,), iterations=5, rounds=5)

    assert(g.title == 'Large')
    assert(len(g.nodes) == nn)
    assert(len(g.edges) <= ne)
