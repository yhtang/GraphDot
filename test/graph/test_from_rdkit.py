#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest
from rdkit import Chem
from graphdot.graph import Graph


@pytest.mark.parametrize('smi', ['CC', 'C=C', 'C#C', 'C1CCCCC1', 'CO'])
def test_from_smiles(smi):
    mol = Chem.MolFromSmiles(smi)
    g = Graph.from_rdkit(mol)
    assert(len(g.nodes) == mol.GetNumAtoms())
