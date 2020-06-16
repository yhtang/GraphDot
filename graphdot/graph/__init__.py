#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""GraphDot's native graph container class

This module defines the class ``Graph`` that are used to store graphs across
this library, and provides conversion and importing methods from popular
graph formats.
"""
import itertools as it
import numpy as np
from graphdot.codegen.typetool import common_min_type
from graphdot.minipandas import DataFrame
from ._from_ase import _from_ase
from ._from_networkx import _from_networkx
from ._from_pymatgen import _from_pymatgen
from ._from_rdkit import _from_rdkit


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
        '''Check if all graphs have the same set of nodal/edge features.'''
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
            features. The types for each attribute will then be
            chosen to be the smallest scalar type that can safely hold all the
            values as found across the graphs.
        inplace: bool
            Whether or not to modify the graph features in-place.

        Returns
        -------
        None or list
            If inplace is True, the graphs will be modified in-place and
            nothing will be returned. Otherwise, a new list of graphs with
            type-unified features will be returned.
        '''

        '''copy graphs if not editing in-place'''
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

        '''ensure all graphs have the same node and edge features'''
        features = {}
        for component in ['nodes', 'edges']:
            first = None
            for g in graphs:
                second = set(getattr(g, component).columns)
                first = first or second
                if second != first:
                    raise TypeError(
                        f'Graph {g} with node features {second} '
                        'does not match with the other graphs.'
                    )
            features[component] = first

        '''unify data type for each feature'''
        for component in ['nodes', 'edges']:
            group = [getattr(g, component) for g in graphs]
            for key in features[component]:
                types = [g[key].concrete_type for g in group]
                t = common_min_type.of_types(types)
                if t == np.object:
                    t = common_min_type.of_types(types, coerce=False)
                if t is None:
                    raise TypeError(
                        f'Cannot unify attribute {key} containing mixed '
                        'object types'
                    )

                if np.issctype(t):
                    for g in group:
                        g[key] = g[key].astype(t)
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

        '''only returns if not editing in-place'''
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
            an undirected graph with homogeneous node and edge features, i.e.
            carrying same features.
        weight: str
            name of the attribute that encode edge weights

        Returns
        -------
        graphdot.graph.Graph
            the converted graph
        """
        return _from_networkx(cls, graph, weight)

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
        return _from_ase(cls, atoms, use_charge, use_pbc, adjacency)

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
            A molecular graph where atoms become nodes while edges resemble
            short-range interatomic interactions.
        """
        return _from_pymatgen(cls, molecule, use_pbc, adjacency)

    @classmethod
    def from_smiles(cls, smiles):
        """DEPRECATED and replaced by from_rdkit."""
        raise RuntimeError(
            'from_smiles has been removed, use from_rdkit instead.'
        )

    @classmethod
    def from_rdkit(cls, mol, bond_type='order', set_ring_list=True,
                   set_ring_stereo=True):
        """Convert a RDKit molecule to a graph

        Parameters
        ----------
        bond_type: 'order' or 'type'
            If 'order', an edge attribute 'order' will be populated with
            numeric values such as 1 for single bonds, 2 for double bonds, and
            1.5 for aromatic bonds. If 'type', an attribute 'type' will be
            populated with :py:class:`rdkit.Chem.BondType` values.
        set_ring_list: bool
            if True, a nodal attribute 'ring_list' will be used to store a list
            of the size of the rings that the atom participates in.
        set_ring_stereo: bool
            If True, an edge attribute 'ring_stereo' will be used to store the
            E-Z stereo configuration of substitutes at the end of each bond
            along a ring.

        Returns
        -------
        graphdot.Graph:
            A graph where nodes represent atoms and edges represent bonds. Each
            node and edge carries an array of features as inferred from the
            chemical structure of the molecule.
        """
        return _from_rdkit(cls, mol,
                           bond_type=bond_type,
                           set_ring_list=set_ring_list,
                           set_ring_stereo=set_ring_stereo)

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
