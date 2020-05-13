#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""GraphDot's native graph container class

This module defines the class ``Graph`` that are used to store graphs across
this library, and provides conversion and importing methods from popular
graph formats.
"""
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
            a molecular graph where atoms become nodes while edges resemble
            short-range interatomic interactions.
        """
        return _from_pymatgen(cls, molecule, use_pbc, adjacency)

    @classmethod
    def from_smiles(cls, smiles):
        """DEPRECATED and superceded by from_rdkit."""
        raise RuntimeError(
            'from_smiles has been deprecated, use from_rdkit instead.'
        )

    @classmethod
    def from_rdkit(cls, mol):
        """Convert a RDKit molecule to a graph"""
        return _from_rdkit(cls, mol)

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
