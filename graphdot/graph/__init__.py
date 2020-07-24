#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""GraphDot's native graph container class

This module defines the class ``Graph`` that are used to store graphs across
this library, and provides conversion and importing methods from popular
graph formats.
"""
import itertools as it
import copy as cp
import numpy as np
import scipy.sparse
import warnings
from graphdot.codegen.typetool import common_min_type
from graphdot.minipandas import DataFrame
from graphdot.util.cookie import VolatileCookie
from ._from_ase import _from_ase
from ._from_networkx import _from_networkx
from ._from_pymatgen import _from_pymatgen
try:
    from ._from_rdkit import _from_rdkit
except ImportError:
    warnings.warn(
        'Cannot import RDKit, `graph.from_rdkit()` will be unavailable.\n'
    )


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
        assert('!i' in self.nodes)
        assert('!i' in self.edges and '!j' in self.edges)

    def __repr__(self):
        return '<{}(nodes={}, edges={}, title={})>'.\
            format(type(self).__name__,
                   repr(self.nodes),
                   repr(self.edges),
                   repr(self.title))

    @property
    def cookie(self):
        try:
            return self.__cookie
        except AttributeError:
            self.__cookie = VolatileCookie()
            return self.__cookie

    def copy(self, deep=False):
        '''Make a copy of an existing graph.

        Parameters
        ----------
        deep: boolean
            If deep=True, then real copies will be made for the node and edge
            dataframes as well as other user-specified attributes. Otherwise,
            only references to the dataframe columns and user-specified
            attributes will be inserted into the new graph.

        Returns
        -------
        g: Graph
            A new graph.
        '''
        g = self.__class__(
            nodes=self.nodes.copy(deep=deep),
            edges=self.edges.copy(deep=deep),
            title=self.title
        )
        for key, val in self.__dict__.items():
            if key not in ['nodes', 'edges', 'title']:
                g.__dict__[key] = cp.deepcopy(val) if deep else val
        return g

    def permute(self, perm, inplace=False):
        '''Rearrange the node indices of a graph by a permutation array.

        Parameters
        ----------
        perm: sequence
            Array of permuted node indices
        inplace: boolean
            Whether to reorder the nodes in-place or to create a new graph.

        Returns
        -------
        permuted_graph: Graph
            The original graph object (inplace=True) or a new one
            (inplace=False) with the nodes permuted.
        '''
        if inplace:
            g = self
            self.cookie.clear()
        else:
            g = self.copy(deep=True)

        iperm = np.argsort(perm)
        g.nodes['!i'][:] = iperm[g.nodes['!i']]
        g.edges['!i'][:] = iperm[g.edges['!i']]
        g.edges['!j'][:] = iperm[g.edges['!j']]

        return g

    @property
    def adjacency_matrix(self):
        '''Get the adjacency matrix of the graph as a sparse matrix.

        Returns
        -------
        adjacency_matrix: sparse matrix
            The adjacency matrix, either weighted or unweighted depending on
            the original graph.
        '''
        N = len(self.nodes)
        i = self.edges['!i']
        j = self.edges['!j']
        w = self.edges['!w'] if '!w' in self.edges else np.ones_like(i)
        A = scipy.sparse.coo_matrix((w, (i, j)), shape=(N, N))
        return A + A.T

    @property
    def laplacian(self):
        '''Get the graph Laplacian as a sparse matrix.

        Returns
        -------
        laplacian: sparse matrix
            The laplacian matrix, either weighted or unweighted depending on
            the original graph.
        '''
        A = self.adjacency_matrix
        D = A.sum(axis=0).flat
        return scipy.sparse.diags(D, 0) - A

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
        for g in graphs:
            g.cookie.clear()
        if inplace is not True:
            graphs = [g.copy(deep=False) for g in graphs]

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
