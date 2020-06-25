#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest
import numpy as np
import itertools as it
import networkx as nx
from graphdot.graph import Graph


def test_empty_init():
    G = Graph(nodes={}, edges={})

    for g in [G, eval(repr(G).strip('><'))]:
        assert(g.title == '')
        assert(len(g.nodes) == 0)
        assert(len(g.nodes.columns) == 0)
        assert(len(g.edges) == 0)
        assert(len(g.edges.columns) == 0)


def test_dict_init():
    G = Graph(nodes={'!i': [0, 1], 'order': [1, -2],
                     'conjugate': [True, False]},
              edges={'!i': [0], '!j': [1], 'length': [3.2],
                     'weight': [1]},
              title='graph')

    for g in [G, eval(repr(G).strip('><'))]:
        assert(g.title == 'graph')
        assert(len(g.nodes) == 2)
        assert(len(g.nodes.columns) == 3)  # +1 for the hidden !i index
        assert(len(g.edges) == 1)
        assert(len(g.edges.columns) == 4)  # +2 for the hidden !i, !j index


@pytest.mark.parametrize('deep', [True, False])
def test_copy(deep):
    nxg = nx.Graph(title='Simple')
    nxg.add_node(0, f=0, g='a')
    nxg.add_node(1, f=1, g='b')
    nxg.add_node(2, f=2, g='c')
    nxg.add_edge(0, 1, h=1.0)
    nxg.add_edge(0, 2, h=2.0)

    g1 = Graph.from_networkx(nxg)
    g1.custom_attributes = (1.5, 'hello', False)
    g2 = g1.copy(deep=deep)
    assert(hasattr(g2, 'custom_attributes'))
    assert(g1.custom_attributes == g2.custom_attributes)

    assert(g1.title == g2.title)
    assert(len(g1.nodes) == len(g2.nodes))
    assert(len(g1.edges) == len(g2.edges))
    for n1, n2 in zip(g1.nodes.rows(), g2.nodes.rows()):
        assert(n1 == n2)
    for e1, e2 in zip(g1.edges.rows(), g2.edges.rows()):
        assert(e1 == e2)

    # Changes in one graph should reflect in its shallow copy, but should not
    # affect its deep copy.
    for i in range(len(g1.nodes)):
        g1.nodes.f[i] += 10.0
    for n1, n2 in zip(g1.nodes.rows(), g2.nodes.rows()):
        if deep:
            assert(n1 != n2)
        else:
            assert(n1 == n2)


@pytest.mark.parametrize('inplace', [True, False])
def test_permute(inplace):
    nxg = nx.Graph(title='Simple')
    f = [np.pi, np.e, np.sqrt(2), -1.0, 2.0]
    nxg.add_node(0, f=f[0])
    nxg.add_node(1, f=f[1])
    nxg.add_node(2, f=f[2])
    nxg.add_node(3, f=f[3])
    nxg.add_node(4, f=f[4])
    nxg.add_edge(0, 1, h=f[0] * f[1])
    nxg.add_edge(0, 2, h=f[0] * f[2])
    nxg.add_edge(0, 4, h=f[0] * f[4])
    nxg.add_edge(1, 2, h=f[1] * f[2])
    nxg.add_edge(1, 3, h=f[1] * f[3])
    nxg.add_edge(2, 3, h=f[2] * f[3])
    nxg.add_edge(3, 4, h=f[3] * f[4])

    g = Graph.from_networkx(nxg)

    for perm in it.permutations(range(len(g.nodes))):
        g1 = g.copy(deep=True)
        g2 = g1.permute(perm, inplace=inplace)

        if inplace:
            assert(g1 is g2)

        m = {idx: i for i, idx in enumerate(g2.nodes['!i'])}
        for i in range(len(g2.edges)):
            i1 = m[g2.edges['!i'][i]]
            i2 = m[g2.edges['!j'][i]]
            assert(g2.edges.h[i] ==
                   pytest.approx(g2.nodes.f[i1] * g2.nodes.f[i2]))


def test_simple_from_networkx():
    nxg = nx.Graph(title='Simple')
    nxg.add_node(0)
    nxg.add_node('c')
    nxg.add_node('X')
    nxg.add_node(3.14)
    nxg.add_edge(0, 'c')
    nxg.add_edge(3.14, 'X')

    G = Graph.from_networkx(nxg)

    for g in [G, eval(repr(G).strip('><'))]:
        assert(g.title == 'Simple')
        assert(len(g.nodes) == 4)
        assert(len(g.nodes.columns) == 1)  # +1 for the hidden !i index
        assert(len(g.edges) == 2)
        assert(len(g.edges.columns) == 2)  # +2 for the hidden !i, !j index


def test_weighted_from_networkx():
    nxg = nx.Graph(title='Simple')
    nxg.add_node(0)
    nxg.add_node(1)
    nxg.add_edge(0, 1, w=1.0)

    G = Graph.from_networkx(nxg, weight='w')

    for g in [G, eval(repr(G).strip('><'))]:
        assert(g.title == 'Simple')
        assert(len(g.nodes) == 2)
        assert(len(g.nodes.columns) == 1)  # +1 for the hidden !i index
        assert(len(g.edges) == 1)
        assert(len(g.edges.columns) == 3)  # +2 for the hidden !i, !j index
        assert('!i' in g.edges.columns)
        assert('!j' in g.edges.columns)
        assert('!w' in g.edges.columns)


def test_molecule_from_networkx():
    nxg = nx.Graph(title='H2O')
    nxg.add_node('O1', charge=1, conjugate=False, mass=8.0)
    nxg.add_node('H1', charge=-1, conjugate=True, mass=1.0)
    nxg.add_node('H2', charge=2, conjugate=True, mass=1.0)
    nxg.add_edge('O1', 'H1', order=1, length=0.5, weight=3.2)
    nxg.add_edge('O1', 'H2', order=2, length=1.0, weight=0.5)

    G = Graph.from_networkx(nxg)

    for g in [G, eval(repr(G).strip('><'))]:
        assert(g.title == 'H2O')
        assert(len(g.nodes) == 3)
        assert(len(g.nodes.columns) == 4)  # +1 for the hidden !i index
        assert(len(g.edges) == 2)
        assert(len(g.edges.columns) == 3+2)  # +2 for the hidden !i, !j index


def test_attribute_consistency_from_networkx():
    nxg1 = nx.Graph(title='H2O')
    nxg1.add_node(0, label1=3.14)
    nxg1.add_node(1, label2=True)

    with pytest.raises(TypeError):
        Graph.from_networkx(nxg1)

    nxg2 = nx.Graph(title='H2O')
    nxg2.add_node(0)
    nxg2.add_node(1)
    nxg2.add_node(2)
    nxg2.add_edge(0, 1, weight=1)
    nxg2.add_edge(0, 2, multi=2)

    with pytest.raises(TypeError):
        Graph.from_networkx(nxg2)


def test_large_from_networkx():
    nxg = nx.Graph(title='Large')
    nn = 1000
    ne = 100000
    np.random.seed(0)
    for i in range(nn):
        nxg.add_node(i, label=np.random.randn())
    for _ in range(ne):
        i, j = np.random.randint(nn, size=2)
        nxg.add_edge(i, j, weight=np.random.rand())

    G = Graph.from_networkx(nxg)

    for g in [G, eval(repr(G).strip('><'))]:
        assert(g.title == 'Large')
        assert(len(g.nodes) == nn)
        assert(len(g.edges) <= ne)
