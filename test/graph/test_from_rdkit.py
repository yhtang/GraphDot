#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest
from rdkit import Chem
from graphdot.graph import Graph


def test_from_rdkit_options():
    m = Chem.MolFromSmiles('CCOCC')
    g = Graph.from_rdkit(m, bond_type='order')
    assert('order' in g.edges.columns)
    g = Graph.from_rdkit(m, bond_type='type')
    assert('type' in g.edges.columns)
    g = Graph.from_rdkit(m, set_ring_list=True)
    assert('ring_list' in g.nodes.columns)
    g = Graph.from_rdkit(m, set_ring_list=False)
    assert('ring_list' not in g.nodes.columns)
    g = Graph.from_rdkit(m, set_ring_stereo=True)
    assert('ring_stereo' in g.edges.columns)
    g = Graph.from_rdkit(m, set_ring_stereo=False)
    assert('ring_stereo' not in g.edges.columns)


def test_from_rdkit_linear_hydrocarbon():
    for i in range(2, 10):
        smi = 'C' * i
        mol = Chem.MolFromSmiles(smi)
        g = Graph.from_rdkit(mol)
        assert(len(g.nodes) == i)
        assert(len(g.edges) == i - 1)


@pytest.mark.parametrize('testset', [
    ('CC', 6),
    ('NN', 7),
    ('OO', 8),
])
def test_from_rdkit_feature_atomic_number(testset):
    smi, atomic_number = testset
    g = Graph.from_rdkit(Chem.MolFromSmiles(smi))
    for n in g.nodes.rows():
        assert(n.atomic_number == atomic_number)


@pytest.mark.parametrize('testset', [
    ('[C+]C', 1),
    ('[C++]C', 2),
    ('[C-]C', -1),
    ('[C--]C', -2),
    ('[C+2]C', 2),
    ('[C+3]C', 3),
    ('[C+4]C', 4),
    ('[C-2]C', -2),
    ('[C-3]C', -3),
])
def test_from_rdkit_feature_atom_charge(testset):
    smi, charge = testset
    g = Graph.from_rdkit(Chem.MolFromSmiles(smi))
    assert(g.nodes.charge[0] == charge)


@pytest.mark.parametrize('testset', [
    ('CC', 3),
    ('C=C', 2),
    ('C#C', 1),
])
def test_from_rdkit_feature_atom_hcount(testset):
    smi, hcount = testset
    g = Graph.from_rdkit(Chem.MolFromSmiles(smi))
    for n in g.nodes.rows():
        assert(n.hcount == hcount)


@pytest.mark.parametrize('testset', [
    ('CC', Chem.HybridizationType.SP3),
    ('C=C', Chem.HybridizationType.SP2),
    ('C#C', Chem.HybridizationType.SP),
])
def test_from_rdkit_feature_atom_hybridization(testset):
    smi, hybridization = testset
    g = Graph.from_rdkit(Chem.MolFromSmiles(smi))
    for n in g.nodes.rows():
        assert(n.hybridization == hybridization)


@pytest.mark.parametrize('testset', [
    ('C1CCCCC1', False),
    ('c1ccccc1', True),
    ('C1=CC=CC=C1', True),
])
def test_from_rdkit_feature_atom_aromatic(testset):
    smi, aromatic = testset
    g = Graph.from_rdkit(Chem.MolFromSmiles(smi))
    for n in g.nodes.rows():
        assert(n.aromatic == aromatic)


@pytest.mark.parametrize('testset', [
    ('[C@H](F)(Cl)Br', Chem.ChiralType.CHI_TETRAHEDRAL_CW),
    ('[C@@H](F)(Cl)Br', Chem.ChiralType.CHI_TETRAHEDRAL_CCW),
])
def test_from_rdkit_feature_atom_chiral(testset):
    smi, chiral = testset
    g = Graph.from_rdkit(Chem.MolFromSmiles(smi))
    assert(g.nodes.chiral[0] == chiral)


@pytest.mark.parametrize('testset', [
    ('CC', [0]),
    ('CCNC', [0]),
    ('COC', [0]),
    ('C1CC1', [3]),
    ('C1CCC1', [4]),
    ('C1CCCC1', [5]),
    ('C1CCCCC1', [6]),
    ('c1ccccc1', [6]),
    ('C1=CC=CC=C1', [6]),
    ('C12(CCC2)CC1', [3, 4]),
])
def test_from_rdkit_feature_atom_ring_list(testset):
    smi, ring_list = testset
    g = Graph.from_rdkit(Chem.MolFromSmiles(smi))
    assert(set(g.nodes.ring_list[0]) == set(ring_list))


@pytest.mark.parametrize('testset', [
    ('CC', 1.0),
    ('C=C', 2.0),
    ('C#C', 3.0),
    ('c1ccccc1', 1.5),
])
def test_from_rdkit_feature_bond_order(testset):
    smi, order = testset
    g = Graph.from_rdkit(Chem.MolFromSmiles(smi))
    for n in g.edges.rows():
        assert(n.order == order)


@pytest.mark.parametrize('testset', [
    ('CC', False),
    ('C=C', False),
    ('C#C', False),
    ('c1ccccc1', True),
    ('C1=CC=CC=C1', True),
])
def test_from_rdkit_feature_bond_aromatic(testset):
    smi, aromatic = testset
    g = Graph.from_rdkit(Chem.MolFromSmiles(smi))
    for n in g.edges.rows():
        assert(n.aromatic == aromatic)


@pytest.mark.parametrize('testset', [
    ('CC', False),
    ('C=C', False),
    ('C=C-C=C', True),
    ('C#C-C#C', True),
    ('C#C', False),
    ('c1ccccc1', True),
    ('C1=CC=CC=C1', True),
])
def test_from_rdkit_feature_bond_conjugated(testset):
    smi, conjugated = testset
    g = Graph.from_rdkit(Chem.MolFromSmiles(smi))
    for n in g.edges.rows():
        assert(n.conjugated == conjugated)


@pytest.mark.parametrize('testset', [
    (r'F/C=C/F', Chem.BondStereo.STEREOE),
    (r'F\C=C/F', Chem.BondStereo.STEREOZ),
    (r'F\C=C\F', Chem.BondStereo.STEREOE),
    (r'F/C=C\F', Chem.BondStereo.STEREOZ),
])
def test_from_rdkit_feature_bond_stereo(testset):
    smi, stereo = testset
    g = Graph.from_rdkit(Chem.MolFromSmiles(smi))
    assert(g.edges.stereo[1] == stereo)
