#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""GraphDot's native graph container class

This module defines the class ``Graph`` that are used to store graphs across
this library, and provides conversion and importing methods from popular
graph formats.
"""
import pandas as pd

__all__ = ['Graph']


def _from_dict(d):
    if isinstance(d, pd.DataFrame):
        return d
    elif all([key in d for key in ['index', 'columns', 'data']]):
        # format of pandas.DataFrame.to_dict('split')
        return pd.DataFrame(d['data'],
                            index=d['index'],
                            columns=d['columns'])
    else:
        # format of pandas.DataFrame.to_dict('dict') i.e. the default style
        return pd.DataFrame(d)


class Graph:
    """
    This is the class that stores a graph in GraphDot.

    Parameters
    ----------
    nodes: dataframe
        each row represent a node
    edges: dataframe
        each row represent an edge
    title: str
        a unique identifier of the graph
    """

    def __init__(self, nodes, edges, title=''):
        self.title = title
        self.nodes = _from_dict(nodes)
        self.edges = _from_dict(edges)

    def __repr__(self):
        return '<{}(nodes={}, edges={}, title={})>'.\
            format(type(self).__name__,
                   self.nodes.to_dict('split'),
                   self.edges.to_dict('split'),
                   repr(self.title))

    # @classmethod
    # def from_auto(cls, graph):
    #     # import importlib
    #     # graph_translator = {}
    #     #
    #     # if importlib.util.find_spec('ase') is not None:
    #     #     ase = importlib.import_module('ase')
    #     #
    #     #     def ase_translator(atoms):
    #     #         pass
    #     #
    #     #     graph_translator[ase.atoms.Atoms] = ase_translator
    #     #
    #     # if importlib.util.find_spec('networkx') is not None:
    #     #     nx = importlib.import_module('networkx')
    #     #
    #     #     def networkx_graph_translator(atoms):
    #     #         pass
    #     #
    #     #     graph_translator[nx.Graph] = networkx_graph_translator
    #     pass

    # @classmethod
    # def from_pymatgen(cls, molecule):
    #     pass

    # @classmethod
    # def from_smiles(cls, smiles):
    #     pass

    @classmethod
    def from_networkx(cls, graph, weight=None):
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
        import networkx as nx
        graph = nx.relabel.convert_node_labels_to_integers(graph)

        ''' extrac title '''
        title = graph.graph['title'] if 'title' in graph.graph.keys() else ''

        ''' convert node attributes '''
        node_attr = []
        for index, node in graph.nodes.items():
            if index == 0:
                node_attr = sorted(node.keys())
            elif node_attr != sorted(node.keys()):
                raise TypeError('Node {} '.format(index) +
                                'attributes {} '.format(node.keys()) +
                                'inconsistent with {}'.format(node_attr))

        node_df = pd.DataFrame(index=range(graph.number_of_nodes()))
        for key in node_attr:
            node_df[key] = [node[key] for node in graph.nodes.values()]

        ''' convert edge attributes '''
        edge_attr = []
        for index, ((i, j), edge) in enumerate(graph.edges.items()):
            if index == 0:
                edge_attr = sorted(edge.keys())
            elif edge_attr != sorted(edge.keys()):
                raise TypeError('Edge {} '.format((i, j)) +
                                'attributes {} '.format(edge.keys()) +
                                'inconsistent with {}'.format(edge_attr))

        edge_df = pd.DataFrame(index=range(graph.number_of_edges()))
        edge_df['!ij'] = list(graph.edges.keys())
        if weight is not None:
            edge_df['!w'] = [edge[weight] for edge in graph.edges.values()]
        for key in edge_attr:
            if key != weight:
                edge_df[key] = [edge[key] for edge in graph.edges.values()]

        return cls(nodes=node_df, edges=edge_df, title=title)

    @classmethod
    def from_molecule(cls, molecule, use_pbc=True, adjacency='default'):
        """Convert molecules to graphs

        Parameters
        ----------
        atoms: an ASE Atoms or pymatgen Molecule object
            A molecule as represented by a collection of atoms in 3D space.
        usb_pbc: boolean or list of 3 booleans
            Whether to use the periodic boundary condition as specified in the
            atoms object to create edges between atoms.
        adjacency: 'default' or object
            A functor that implements the rule for making edges between atoms.

        Returns
        -------
        Graph:
            a molecular graph where atoms become nodes while edges resemble
            short-range interatomic interactions.
        """
        raise RuntimeError('To convert from molecules, import graph.molecular')

    # @classmethod
    # def from_graphviz(cls, molecule):
    #     pass
    #
    # @classmethod
    # def from_dot(cls, molecule):
    #     """
    #     From the DOT graph description language
    #     https://en.wikipedia.org/wiki/DOT_(graph_description_language)
    #     """
    #     pass
