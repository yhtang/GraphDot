#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest
from graphdot.graph import Graph


def test_to_networkx():
    graph = Graph(
        {
            '!i': [0, 1, 2],
            'open': [True, False, True]
        },
        {
            '!i': [0, 0],
            '!j': [1, 2],
            'length': [0.5, 1.5],
            'label': ['x', 'y']
        },
        title='test'
    )

    ngraph = graph.to_networkx()

    assert(len(ngraph.nodes) == len(graph.nodes))
    assert(len(ngraph.edges) == len(graph.edges))
    for _, n in ngraph.nodes.items():
        assert('open' in n)
    for _, e in ngraph.edges.items():
        assert('length' in e)
        assert('label' in e)
    for k in range(len(graph.nodes)):
        i = graph.nodes['!i'][k]
        assert(graph.nodes['open'][k] == ngraph.nodes[i]['open'])
    for k in range(len(graph.edges)):
        i = graph.edges['!i'][k]
        j = graph.edges['!j'][k]
        assert(graph.edges['length'][k] == ngraph[i][j]['length'])
        assert(graph.edges['label'][k] == ngraph[i][j]['label'])
