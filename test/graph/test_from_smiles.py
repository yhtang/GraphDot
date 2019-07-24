#!/usr/bin/env python
# -*- coding: utf-8 -*-
import re
import pytest
from graphdot.graph import Graph


@pytest.mark.parametrize('smi', ['C', 'CC', 'CCC', 'CCCC', 'C=C', 'C=C-C',
                                 'C=C-C=C', 'CN', 'CCN', 'C=CN', 'CC=N'])
def test_linear(smi):
    g = Graph.from_smiles(smi)
    assert(len(g.nodes) == len(re.sub('[^CNOH]', '', smi)))
    assert(len(g.edges) == len(g.nodes) - 1)


@pytest.mark.parametrize('smi', ['C1CC1', 'C1CCC1', 'C1CCCC1'])
def test_cyclic(smi):
    g = Graph.from_smiles(smi)
    assert(len(g.nodes) == len(re.sub('[^CNOH]', '', smi)))
    assert(len(g.edges) == len(g.nodes))
