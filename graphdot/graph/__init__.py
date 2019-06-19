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

        pass

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
    def from_networkx(cls, graph, check=True):
        """
        check: whether to perform node/edge attribute consistency check
        """
        import networkx as nx
        graph = nx.relabel.convert_node_labels_to_integers(graph)

        ''' convert nodes attributes '''
        attr_keys = sorted(graph.nodes[0])
        if check:
            for index, attributes in graph.nodes.items():
                if sorted(attributes.keys()) != attr_keys:
                    raise TypeError('Node {} attribute(s) {} inconsistent with {}'.
                                    format(index, attributes.keys(), attr_keys))
        node_attr = [graph.nodes[index].values() for index in graph.nodes]

        nodes = pandas.DataFrame(node_attr, columns=attr_keys)

        print(nodes)

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
        SP   = 1
        SP2  = 2
        SP3  = 3

    g = nx.Graph()
    g.add_node('O1', hybridization=Hybridization.SP2, charge=1)
    g.add_node('H1', hybridization=Hybridization.SP3, charge=-1)
    # g.add_node('H2', hybridization=Hybridization.SP, charge=2)
    g.add_node('H2', hybridization=Hybridization.SP, charge=2, time=1)
    g.add_edge('O1', 'H1', order=1, length=0.5)
    g.add_edge('O1', 'H2', order=2, length=1.0)

    Graph.from_networkx(g)
