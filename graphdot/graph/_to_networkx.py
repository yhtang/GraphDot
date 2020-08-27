#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Convertor to NetworkX's Graph objects"""
import networkx as nx


def _to_networkx(graph):
    """Convert to NetworkX ``Graph``.

    Parameters
    ----------
    graph: GraphDot graph instance
        The graph to be converted

    Returns
    -------
    networkx.Graph
        the converted graph
    """

    nxgraph = nx.from_pandas_edgelist(
        graph.edges,
        source='!i',
        target='!j',
        edge_attr=True
    )

    nxgraph.graph['title'] = graph.title

    nx.set_node_attributes(
        nxgraph,
        {i: r._asdict() for i, r in graph.nodes.iterrows()}
    )

    return nxgraph
