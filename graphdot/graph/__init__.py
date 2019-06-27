#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This package defines the Graph container class of graphdot and converts from
popular graph libraries.
"""
import pandas as pd


class Graph(object):

    def __init__(self, nodes, edges, title=''):
        """
        nodes: dataframe
        edges: dataframe
        """
        self.title = title
        if isinstance(nodes, pd.DataFrame):
            self.nodes = nodes
        else:
            self.nodes = pd.DataFrame(nodes)
        if isinstance(edges, pd.DataFrame):
            self.edges = edges
        else:
            self.edges = pd.DataFrame(edges)

    def __repr__(self):
        return '<{}(nodes={}, edges={}, title={})>'.\
            format(type(self).__name__,
                   self.nodes.to_dict(),
                   self.edges.to_dict(),
                   repr(self.title))

    @classmethod
    def from_auto(cls, graph):
        # import importlib
        # graph_translator = {}
        #
        # if importlib.util.find_spec('ase') is not None:
        #     ase = importlib.import_module('ase')
        #
        #     def ase_translator(atoms):
        #         pass
        #
        #     graph_translator[ase.atoms.Atoms] = ase_translator
        #
        # if importlib.util.find_spec('networkx') is not None:
        #     nx = importlib.import_module('networkx')
        #
        #     def networkx_graph_translator(atoms):
        #         pass
        #
        #     graph_translator[nx.Graph] = networkx_graph_translator
        pass

    @classmethod
    def from_ase(cls, atoms):
        pass

    @classmethod
    def from_pymatgen(cls, molecule):
        pass

    @classmethod
    def from_networkx(cls, graph):
        """
        graph: NetworkX Graph objects with nodal/edge attributes
        """
        import networkx as nx
        graph = nx.relabel.convert_node_labels_to_integers(graph)

        ''' extrac title '''
        title = graph.graph['title'] if 'title' in graph.graph.keys() else ''

        ''' convert node attributes '''
        node_attr = None
        for index, node in graph.nodes.items():
            if node_attr is None:
                node_attr = sorted(node.keys())
            elif node_attr != sorted(node.keys()):
                raise TypeError('Node {} '.format(index) +
                                'attributes {} '.format(node.keys()) +
                                'inconsistent with {}'.format(node_attr))

        node_df = pd.DataFrame(index=range(graph.number_of_nodes()))
        for key in node_attr:
            node_df[key] = [node[key] for node in graph.nodes.values()]

        ''' convert edge attributes '''
        edge_attr = None
        for (i, j), edge in graph.edges.items():
            if edge_attr is None:
                edge_attr = sorted(edge.keys())
            elif edge_attr != sorted(edge.keys()):
                raise TypeError('Edge {} '.format((i, j)) +
                                'attributes {} '.format(edge.keys()) +
                                'inconsistent with {}'.format(edge_attr))

        edge_df = pd.DataFrame(index=range(graph.number_of_edges()))
        edge_df['!ij'] = list(graph.edges.keys())
        for key in edge_attr:
            edge_df[key] = [edge[key] for edge in graph.edges.values()]

        return cls(nodes=node_df, edges=edge_df, title=title)

    @classmethod
    def from_graphviz(cls, molecule):
        pass

    @classmethod
    def from_dot(cls, molecule):
        """
        From the DOT graph description language
        https://en.wikipedia.org/wiki/DOT_(graph_description_language)
        """
        pass


if __name__ == '__main__':

    import networkx as nx

    class Hybrid:
        NONE = 0
        SP = 1
        SP2 = 2
        SP3 = 3

    g = nx.Graph(title='H2O')
    g.add_node('O1', hybridization=Hybrid.SP2, charge=1, conjugate=False)
    g.add_node('H1', hybridization=Hybrid.SP3, charge=-1, conjugate=True)
    g.add_node('H2', hybridization=Hybrid.SP, charge=2, conjugate=True)
    g.add_edge('O1', 'H1', order=1, length=0.5)
    g.add_edge('O1', 'H2', order=2, length=1.0)

    print(Graph.from_networkx(g))
