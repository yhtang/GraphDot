#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Adaptor for RDKit's Molecule objects"""
import re
import networkx as nx
import numpy as np
from treelib import Tree
from rdkit.Chem import AllChem as Chem
from ._from_networkx import _from_networkx


class FunctionalGroup:
    """Functional Group.

    atom0 -> atom1 define a directed bond in the molecule. Then the bond is
    removed and the functional group is defined as a multitree. atom1 is the
    root node of the tree.

    Parameters
    ----------
    mol : molecule object in RDKit

    atom0, atom1 : atom object in RDKit

    depth: the depth of the multitree.

    Attributes
    ----------
    tree : multitree represent the functional group
        each node has 3 important attributes: tag: [atomic number, bond order
        with its parent], identifier: atom index defined in RDKit molecule
        object, data: RDKit atom object.

    """

    def __init__(self, mol, atom0, atom1, depth=5):
        self.mol = mol
        tree = Tree()
        bond_order = mol.GetBondBetweenAtoms(
            atom0.GetIdx(),
            atom1.GetIdx()
        ).GetBondTypeAsDouble()
        tree.create_node(
            tag=[atom0.GetAtomicNum(), bond_order],
            identifier=atom0.GetIdx(), data=atom0
        )
        tree.create_node(
            tag=[atom1.GetAtomicNum(), bond_order],
            identifier=atom1.GetIdx(),
            data=atom1,
            parent=atom0.GetIdx()
        )
        for _ in range(depth):
            for node in tree.all_nodes():
                if node.is_leaf():
                    for atom in node.data.GetNeighbors():
                        tree_id = tree._identifier
                        if atom.GetIdx() != node.predecessor(tree_id=tree_id):
                            order = mol.GetBondBetweenAtoms(
                                atom.GetIdx(),
                                node.data.GetIdx()
                            ).GetBondTypeAsDouble()
                            identifier = atom.GetIdx()
                            while tree.get_node(identifier) is not None:
                                identifier += len(mol.GetAtoms())
                            tree.create_node(
                                tag=[atom.GetAtomicNum(), order],
                                identifier=identifier,
                                data=atom,
                                parent=node.identifier
                            )
        self.tree = tree

    def __eq__(self, other):
        if self.get_rank_list() == other.get_rank_list():
            return True
        else:
            return False

    def __lt__(self, other):
        if self.get_rank_list() < other.get_rank_list():
            return True
        else:
            return False

    def __gt__(self, other):
        if self.get_rank_list() > other.get_rank_list():
            return True
        else:
            return False

    def get_rank_list(self):
        rank_list = []
        expand_tree = self.tree.expand_tree(mode=Tree.WIDTH, reverse=True)
        for identifier in expand_tree:
            rank_list += self.tree.get_node(identifier).tag
        return rank_list


def get_bond_orientation_dict(mol):
    bond_orientation_dict = {}
    mb = Chem.MolToMolBlock(mol, includeStereo=True, kekulize=False)
    for i, j, _, d in re.findall(r'^[\s*(\d+)]{4}$', mb, re.MULTILINE):
        i, j, d = int(i) - 1, int(j) - 1, int(d)
        i, j = min(i, j), max(i, j)
        bond_orientation_dict[(i, j)] = d
    return bond_orientation_dict


def get_atom_ring_stereo(mol, atom, ring_idx, depth=5,
                         bond_orientation_dict=None):
    """Return an atom is upward or downward refer to a ring plane.

    For atom in a ring. If it has 4 bonds. Two of them are included in the
    ring. Other two connecting 2 functional groups, has opposite orientation
    reference to the ring plane. Assuming the ring is in a plane, then the 2
    functional groups are assigned as upward and downward.

    Parameters
    ----------
    mol : molecule object in RDKit

    atom : atom object in RDKit

    ring_idx : a tuple of all index of atoms in the ring

    depth : the depth of the functional group tree

    bond_orientation_dict : a dictionary contains the all bond orientation
        information in the molecule

    Returns
    -------
    0 : No ring stereo.

    1 : The upward functional group is larger

    -1 : The downward functional group is larger

    """
    if bond_orientation_dict is None:
        bond_orientation_dict = get_bond_orientation_dict(mol)

    up_atom = down_atom = None
    updown_tag = None
    # bond to 2 hydrogen
    if len(atom.GetNeighbors()) == 2:
        return 0
    if len(atom.GetNeighbors()) > 4:
        raise RuntimeError(
            'cannot deal with atom in a ring with more than 4 bonds'
        )
    for bond in atom.GetBonds():
        # for carbon atom, atom ring stereo may exist if it has 4 single bonds.
        if bond.GetBondType() != Chem.BondType.SINGLE \
                and atom.GetAtomicNum() == 6:
            return 0
        i = bond.GetBeginAtom().GetIdx()
        j = bond.GetEndAtom().GetIdx()
        ij = (i, j)
        # skip bonded atoms in the ring
        if i in ring_idx and j in ring_idx:
            # in RDKit, the orientation information may saved in ring bond for
            # multi-ring molecules. The information is saved.
            if bond_orientation_dict.get(ij) != 0:
                updown_tag = bond_orientation_dict.get(ij)
            continue
        # get upward atom
        if bond_orientation_dict.get(ij) == 1:
            if up_atom is not None:
                raise Exception('2 bond orient up')
            temp = list(ij)
            temp.remove(atom.GetIdx())
            up_atomidx = temp[0]
            up_atom = mol.GetAtomWithIdx(up_atomidx)
        # get downward atom
        elif bond_orientation_dict.get(ij) == 6:
            if down_atom is not None:
                raise Exception('2 bond orient down')
            temp = list(ij)
            temp.remove(atom.GetIdx())
            down_atomidx = temp[0]
            down_atom = mol.GetAtomWithIdx(down_atomidx)
    # maybe there is bug for complex molecule
    if up_atom is None and down_atom is None:
        if updown_tag == 1:
            return 1
        elif updown_tag == 6:
            return -1
        else:
            return 0
    elif up_atom is None:
        return -1
    elif down_atom is None:
        return 1
    else:
        fg_up = FunctionalGroup(mol, atom, up_atom, depth)
        fg_down = FunctionalGroup(mol, atom, down_atom, depth)
        if fg_up > fg_down:
            return 1
        elif fg_up < fg_down:
            return -1
        else:
            return 0


def get_ringlist(mol):
    ringlist = [[] for _ in range(mol.GetNumAtoms())]
    for ring in mol.GetRingInfo().AtomRings():
        for i in ring:
            ringlist[i].append(len(ring))
    return [sorted(rings) if len(rings) else [0] for rings in ringlist]


def _from_rdkit(cls, mol, bond_type='order', set_ring_list=True,
                set_ring_stereo=True):
    g = nx.Graph()

    for i, atom in enumerate(mol.GetAtoms()):
        g.add_node(i)
        g.nodes[i]['atomic_number'] = atom.GetAtomicNum()
        g.nodes[i]['charge'] = atom.GetFormalCharge()
        g.nodes[i]['hcount'] = atom.GetTotalNumHs()
        g.nodes[i]['hybridization'] = atom.GetHybridization()
        g.nodes[i]['aromatic'] = atom.GetIsAromatic()
        g.nodes[i]['chiral'] = 0 if atom.IsInRing() else atom.GetChiralTag()

    if set_ring_list:
        for i, rings in enumerate(get_ringlist(mol)):
            g.nodes[i]['ring_list'] = rings

    for bond in mol.GetBonds():
        ij = (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
        g.add_edge(*ij)
        if bond_type == 'order':
            g.edges[ij]['order'] = bond.GetBondTypeAsDouble()
        else:
            g.edges[ij]['type'] = bond.GetBondType()
        g.edges[ij]['aromatic'] = bond.GetIsAromatic()
        g.edges[ij]['conjugated'] = bond.GetIsConjugated()
        g.edges[ij]['stereo'] = bond.GetStereo()
        if set_ring_stereo is True:
            g.edges[ij]['ring_stereo'] = 0

    if set_ring_stereo is True:
        bond_orientation_dict = get_bond_orientation_dict(mol)
        for ring_idx in mol.GetRingInfo().AtomRings():
            atom_updown = []
            for idx in ring_idx:
                atom = mol.GetAtomWithIdx(idx)
                atom_updown.append(
                    get_atom_ring_stereo(
                        mol,
                        atom,
                        ring_idx,
                        depth=5,
                        bond_orientation_dict=bond_orientation_dict
                    )
                )
            atom_updown = np.array(atom_updown)
            non_zero_index = np.where(atom_updown != 0)[0]
            for j in range(len(non_zero_index)):
                b = non_zero_index[j]
                if j == len(non_zero_index) - 1:
                    e = non_zero_index[0]
                    length = len(atom_updown) + e - b
                else:
                    e = non_zero_index[j + 1]
                    length = e - b
                StereoOfRingBond = atom_updown[b] * atom_updown[e] / length
                for k in range(length):
                    idx1 = b + k if b + k < len(ring_idx)\
                        else b + k - len(ring_idx)
                    idx2 = b + k + 1 if b + k + 1 < len(ring_idx)\
                        else b + k + 1 - len(ring_idx)
                    ij = (ring_idx[idx1], ring_idx[idx2])
                    ij = (min(ij), max(ij))
                    g.edges[ij]['ring_stereo'] = StereoOfRingBond

    return _from_networkx(cls, g)
