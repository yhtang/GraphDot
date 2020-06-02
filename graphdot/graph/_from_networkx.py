#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Adaptor for NetworkX's Graph objects"""
import networkx as nx
from graphdot.minipandas import DataFrame


def _from_networkx(cls, graph, weight=None):
    """Convert from NetworkX ``Graph``

    Parameters
    ----------
    graph: a NetworkX ``Graph`` instance
        an undirected graph with homogeneous node and edge attributes, i.e.
        carrying same attributes.
    weight: str
        name of the attribute that encode edge weights

    Returns
    -------
    graphdot.graph.Graph
        the converted graph
    """

    nodes = list(graph.nodes)

    if not all(isinstance(x, int) for x in nodes) \
            or max(nodes) + 1 != len(nodes) or min(nodes) < 0:
        graph = nx.relabel.convert_node_labels_to_integers(graph)

    ''' extrac title '''
    title = graph.graph['title'] if 'title' in graph.graph.keys() else ''

    ''' convert node attributes '''
    node_attr = []
    for index, node in graph.nodes.items():
        if index == 0:
            node_attr = sorted(node.keys())
        elif node_attr != sorted(node.keys()):
            # raise TypeError(f'Node {index} '
            #                 f'attributes {node.keys()} '
            #                 f'inconsistent with {node_attr}')
            raise TypeError('Node {} attributes {} '
                            'inconsistent with {}'.format(
                                index,
                                node.keys(),
                                node_attr))

    node_df = DataFrame({'!i': range(len(graph.nodes))})
    for key in node_attr:
        node_df[key] = [node[key] for node in graph.nodes.values()]

    ''' convert edge attributes '''
    edge_attr = []
    for index, ((i, j), edge) in enumerate(graph.edges.items()):
        if index == 0:
            edge_attr = sorted(edge.keys())
        elif edge_attr != sorted(edge.keys()):
            # raise TypeError(f'Edge {(i, j)} '
            #                 f'attributes {edge.keys()} '
            #                 f'inconsistent with {edge_attr}')
            raise TypeError('Edge {} attributes {} '
                            'inconsistent with {}'.format(
                                (i, j),
                                edge.keys(),
                                edge_attr
                            ))

    edge_df = DataFrame()
    if len(graph.edges.keys()) == 0:
        raise RuntimeError(f'Graph {graph} has no edges.')
    edge_df['!i'], edge_df['!j'] = zip(*graph.edges.keys())
    if weight is not None:
        edge_df['!w'] = [edge[weight] for edge in graph.edges.values()]
    for key in edge_attr:
        if key != weight:
            edge_df[key] = [edge[key] for edge in graph.edges.values()]

    return cls(nodes=node_df, edges=edge_df, title=title)
