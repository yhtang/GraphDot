#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest
import numpy as np
import networkx as nx
from graphdot.graph import Graph


def test_empty_init():
    G = Graph(nodes=[], edges=[])

    for g in [G, eval(repr(G).strip('><'))]:
        assert(g.title == '')
        assert(len(g.nodes) == 0)
        assert(len(g.nodes.columns) == 0)
        assert(len(g.edges) == 0)
        assert(len(g.edges.columns) == 0)


def test_dict_init():
    G = Graph(nodes={'order': {0: 1, 1: -2}, 'conjugate': {0: True, 1: False}},
              edges={'!ij': {0: (0, 1)}, 'length': {0: 3.2}, 'weight': {0: 1}},
              title='graph')

    for g in [G, eval(repr(G).strip('><'))]:
        assert(g.title == 'graph')
        assert(len(g.nodes) == 2)
        assert(len(g.nodes.columns) == 2)
        assert(len(g.edges) == 1)
        assert(len(g.edges.columns) == 3)


def test_empty_from_networkx():
    nxg = nx.Graph(title='Null')

    G = Graph.from_networkx(nxg)

    for g in [G, eval(repr(G).strip('><'))]:
        assert(g.title == 'Null')
        assert(len(g.nodes) == 0)
        assert(len(g.nodes.columns) == 0)
        assert(len(g.edges) == 0)
        assert(len(g.edges.columns) == 1)  # +1 for the hidden edge index


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
        assert(len(g.nodes.columns) == 0)
        assert(len(g.edges) == 2)
        assert(len(g.edges.columns) == 1)  # +1 for the hidden edge index


def test_weighted_from_networkx():
    nxg = nx.Graph(title='Simple')
    nxg.add_node(0)
    nxg.add_node(1)
    nxg.add_edge(0, 1, w=1.0)

    G = Graph.from_networkx(nxg, weight='w')

    for g in [G, eval(repr(G).strip('><'))]:
        assert(g.title == 'Simple')
        assert(len(g.nodes) == 2)
        assert(len(g.nodes.columns) == 0)
        assert(len(g.edges) == 1)
        assert(len(g.edges.columns) == 2)  # +2 for edge index and weight
        assert('!ij' in g.edges.columns)
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
        assert(len(g.nodes.columns) == 3)
        assert(len(g.edges) == 2)
        assert(len(g.edges.columns) == 3+1)  # +1 for the hidden edge index


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
