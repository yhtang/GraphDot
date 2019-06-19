#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This package defines the Graph container class of graphdot and converts from
popular graph libraries.
"""
import pandas


class Graph:

    def __init__(self, nodes, edges):
        """
        nodes: dataframe
        edges: dataframe
        """
        self.nodes = nodes
        self.edges = edges

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

        ''' convert node attributes '''
        node_list = []
        node_attr = None
        for index, node in graph.nodes.items():
            if node_attr is None:
                node_attr = sorted(node.keys())
            elif node_attr != sorted(node.keys()):
                raise TypeError('Node {} '.format(index) +
                                'attributes {} '.format(node.keys()) +
                                'inconsistent with {}'.format(node_attr))
            node_list.append([node[key] for key in node_attr])
        node_df = pandas.DataFrame(node_list, columns=node_attr)

        ''' convert edge attributes '''
        edge_list = []
        edge_attr = None
        for (i, j), edge in graph.edges.items():
            if edge_attr is None:
                edge_attr = sorted(edge.keys())
            elif edge_attr != sorted(edge.keys()):
                raise TypeError('Edge {} '.format((i, j)) +
                                'attributes {} '.format(edge.keys()) +
                                'inconsistent with {}'.format(edge_attr))
            edge_list.append([i, j] + [edge[key] for key in edge_attr])
        edge_df = pandas.DataFrame(edge_list, columns=['_i', '_j'] + edge_attr)

        return cls(node_df, edge_df)

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

    class Hybridization:
        NONE = 0
        SP = 1
        SP2 = 2
        SP3 = 3

    g = nx.Graph()
    g.add_node('O1', hybridization=Hybridization.SP2, charge=1)
    g.add_node('H1', hybridization=Hybridization.SP3, charge=-1)
    g.add_node('H2', hybridization=Hybridization.SP, charge=2)
    # g.add_node('H2', hybridization=Hybridization.SP, charge=2, time=1)
    g.add_edge('O1', 'H1', order=1, length=0.5)
    g.add_edge('O1', 'H2', order=2, length=1.0)

    gg = Graph.from_networkx(g)
