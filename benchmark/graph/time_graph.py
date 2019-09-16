import pytest
import numpy as np
import networkx as nx
from ase.build import molecule, fcc111
from graphdot.graph import Graph


@pytest.mark.parametrize("n", [10, 100, 1000])
def test_graph_from_networkx(n, benchmark):

    nxg = nx.Graph(title='Large')
    np.random.seed(0)
    for i in range(n):
        nxg.add_node(i, label=np.random.randn())
    for _ in range(n * 10):
        i, j = np.random.randint(n, size=2)
        nxg.add_edge(i, j, weight=np.random.rand())

    def fun(nxg):
        return Graph.from_networkx(nxg)

    g = benchmark.pedantic(fun, args=(nxg,), iterations=5, rounds=5,
                           warmup_rounds=1)

    assert(g.title == 'Large')
    assert(len(g.nodes) == nxg.number_of_nodes())
    assert(len(g.edges) == nxg.number_of_edges())


@pytest.mark.parametrize("atoms", [
    molecule('H2O'),
    molecule('bicyclobutane'),
    fcc111('Cu', size=(4, 4, 4), periodic=True),
    fcc111('Cu', size=(8, 8, 8), periodic=True),
])
def test_graph_from_ase(atoms, benchmark):

    def fun(atoms):
        return Graph.from_ase(atoms)

    g = benchmark.pedantic(fun, args=(atoms,), iterations=5, rounds=5,
                           warmup_rounds=1)

    assert(len(g.nodes) == len(atoms))
