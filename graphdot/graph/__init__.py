#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""GraphDot's native graph container class

This module defines the class ``Graph`` that are used to store graphs across
this library, and provides conversion and importing methods from popular
graph formats.
"""
import uuid
import itertools as it
import numpy as np
from scipy.spatial import cKDTree
from graphdot.codegen.typetool import common_min_type
from graphdot.graph.adjacency.atomic import AtomicAdjacency
from graphdot.minipandas import DataFrame

__all__ = ['Graph']


def _from_dict(d):
    if isinstance(d, DataFrame):
        return d
    else:
        # dict of column name-data pairs
        return DataFrame(d)


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
        self.title = str(title)
        self.nodes = _from_dict(nodes)
        self.edges = _from_dict(edges)

    def __repr__(self):
        return '<{}(nodes={}, edges={}, title={})>'.\
            format(type(self).__name__,
                   repr(self.nodes),
                   repr(self.edges),
                   repr(self.title))

    @staticmethod
    def has_unified_types(graphs):
        '''Check if all graphs have the same set of nodal/edge attributes.'''
        first = next(iter(graphs))
        node_t = first.nodes.rowtype()
        edge_t = first.edges.rowtype()
        for second in graphs:
            if second.nodes.rowtype() != node_t:
                return ('nodes', first, second)
            elif second.edges.rowtype() != edge_t:
                return ('edges', first, second)
        return True

    @classmethod
    def unify_datatype(cls, graphs, inplace=False):
        '''Ensure that each attribute has the same data type across graphs.

        Parameters
        ----------
        graphs: list
            A list of graphs that have the same set of node and edge
            attributes. The types for each attribute will then be
            chosen to be the smallest scalar type that can safely hold all the
            values as found across the graphs.
        inplace: bool
            Whether or not to modify the graph attributes in-place.

        Returns
        -------
        None or list
            If inplace is True, the graphs will be modified in-place and
            nothing will be returned. Otherwise, a new list of graphs with
            type-unified attributes will be returned.
        '''
        if inplace is not True:
            def shallowcopy(g):
                h = cls(
                    nodes=g.nodes.copy(deep=False),
                    edges=g.edges.copy(deep=False),
                    title=g.title
                )
                for key, val in g.__dict__.items():
                    if key not in ['nodes', 'edges', 'title']:
                        h.__dict__[key] = val
                return h
            graphs = [shallowcopy(g) for g in graphs]

        attributes = {}
        for component in ['nodes', 'edges']:
            first = None
            for g in graphs:
                second = set(getattr(g, component).columns)
                first = first or second
                if second != first:
                    raise TypeError(
                        f'Graph {g} with node attributes {second} '
                        'does not match with the other graphs.'
                    )
            attributes[component] = first
        for component in ['nodes', 'edges']:
            group = [getattr(g, component) for g in graphs]
            for key in attributes[component]:
                types = [g[key].get_type(concrete=True) for g in group]
                t = common_min_type.of_types(types)
                if t == np.object:
                    t = common_min_type.of_types(types, coerce=False)
                if t is None:
                    raise TypeError(
                        f'Cannot unify attribute {key} containing mixed '
                        'object types'
                    )

                # print(component, key, t)

                if np.issctype(t):
                    for g in group:
                        g[key] = g[key].astype(t)
                        # print(f'g[{key}].dtype', g[key].dtype)
                elif t in [list, tuple, np.ndarray]:
                    t_sub = common_min_type.of_values(
                        it.chain.from_iterable(
                            it.chain.from_iterable([g[key] for g in group])
                        )
                    )
                    if t_sub is None:
                        raise TypeError(
                            f'Cannot find a common type for elements in {key}.'
                        )
                    for g in group:
                        g[key] = [np.array(seq, dtype=t_sub) for seq in g[key]]

        if inplace is not True:
            return graphs

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
    #     #     graph_translator[ase.atomsatoms.Atoms] = ase_translator
    #     #
    #     # if importlib.util.find_spec('networkx') is not None:
    #     #     nx = importlib.import_module('networkx')
    #     #
    #     #     def networkx_graph_translator(atoms):
    #     #         pass
    #     #
    #     #     graph_translator[nx.Graph] = networkx_graph_translator
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
                raise TypeError(f'Edge {(i, j)} '
                                f'attributes {edge.keys()} '
                                f'inconsistent with {edge_attr}')

        edge_df = DataFrame()
        edge_df['!i'], edge_df['!j'] = zip(*graph.edges.keys())
        if weight is not None:
            edge_df['!w'] = [edge[weight] for edge in graph.edges.values()]
        for key in edge_attr:
            if key != weight:
                edge_df[key] = [edge[key] for edge in graph.edges.values()]

        return cls(nodes=node_df, edges=edge_df, title=title)

    @classmethod
    def from_ase(cls, atoms, use_charge=False, use_pbc=True,
                 adjacency='default'):
        """Convert from ASE atoms to molecular graph

        Parameters
        ----------
        atoms: ASE Atoms object
            A molecule as represented by a collection of atoms in 3D space.
        usb_pbc: boolean or list of 3 booleans
            Whether to use the periodic boundary condition as specified in the
            atoms object to create edges between atoms.
        adjacency: 'default' or object
            A functor that implements the rule for making edges between atoms.

        Returns
        -------
        graphdot.Graph:
            a molecular graph where atoms become nodes while edges resemble
            short-range interatomic interactions.
        """
        if adjacency == 'default':
            adjacency = AtomicAdjacency()

        nodes = DataFrame({'!i': range(len(atoms))})
        nodes['element'] = atoms.get_atomic_numbers().astype(np.int8)
        if use_charge:
            nodes['charge'] = atoms.get_initial_charges().astype(np.float32)

        pbc = np.logical_and(atoms.pbc, use_pbc)
        images = [(atoms.cell.T * image).sum(axis=1) for image in it.product(
            *tuple([-1, 0, 1] if p else [0] for p in pbc))]
        x = atoms.get_positions()
        x_images = np.vstack([x + i for i in images])
        # prefer lookup over integer modulo
        j_images = list(range(len(atoms))) * len(images)

        cutoff = adjacency.cutoff(atoms.get_atomic_numbers())
        nl = cKDTree(x).sparse_distance_matrix(cKDTree(x_images), cutoff)

        edgedict = {}
        for (i, j), r in nl.items():
            j = j_images[j]
            if j > i:
                w = adjacency(atoms[i].number, atoms[j].number, r)
                if w > 0 and ((i, j) not in edgedict or
                              edgedict[(i, j)][1] > r):
                    edgedict[(i, j)] = (w, r)
        i, j, w, r = list(zip(*[(i, j, w, r)
                                for (i, j), (w, r) in edgedict.items()]))

        edges = DataFrame({
            '!i': np.array(i, dtype=np.uint32),
            '!j': np.array(j, dtype=np.uint32),
            '!w': np.array(w, dtype=np.float32),
            'length': np.array(r, dtype=np.float32),
        })

        return cls(nodes, edges, title='Molecule {formula} {id}'.format(
                   formula=atoms.get_chemical_formula(), id=uuid.uuid4().hex))

    @classmethod
    def from_pymatgen(cls, molecule, use_pbc=True, adjacency='default'):
        """Convert from pymatgen molecule to molecular graph

        Parameters
        ----------
        molecule: pymatgen Molecule object
            A molecule as represented by a collection of atoms in 3D space.
        usb_pbc: boolean or list of 3 booleans
            Whether to use the periodic boundary condition as specified in the
            atoms object to create edges between atoms.
        adjacency: 'default' or object
            A functor that implements the rule for making edges between atoms.

        Returns
        -------
        graphdot.Graph:
            a molecular graph where atoms become nodes while edges resemble
            short-range interatomic interactions.
        """
        import pymatgen.io
        atoms = pymatgen.io.ase.AseAtomsAdaptor.get_atoms(molecule)
        return cls.from_ase(atoms, use_pbc, adjacency)

    @classmethod
    def from_smiles(cls, smiles):
        """ Convert from a SMILES string to molecular graph

        Parameters
        ----------
        smiles: str
            A string encoding a molecule using the OpenSMILES format

        Returns
        -------
        graphdot.Graph:
            A molecular graph where atoms becomes nodes with the 'aromatic',
            'charge', 'element', 'hcount' attributes, and bonds become edges
            with the 'order' attribute.
        """
        import pysmiles.read_smiles
        from mendeleev import element
        m = pysmiles.read_smiles(smiles)
        for _, n in m.nodes.items():
            n['element'] = element(n['element']).atomic_number
        graph = cls.from_networkx(m)
        for attr, dtype in zip(['aromatic', 'charge', 'element', 'hcount'],
                               [np.bool_, np.float32, np.int8, np.int8]):
            graph.nodes[attr] = graph.nodes[attr].astype(dtype)
        graph.edges['order'] = graph.edges['order'].astype(np.float32)
        return graph

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
