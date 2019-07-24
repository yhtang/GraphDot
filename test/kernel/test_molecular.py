#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest
import numpy as np
from ase.build import molecule
from graphdot import Graph
from graphdot.kernel.molecular import Tang2019MolecularKernel


def test_molecular_kernel():
    molecules = [molecule('H2'), molecule('O2'), molecule('CH4')]

    graphs = [Graph.from_ase(m) for m in molecules]

    kernel = Tang2019MolecularKernel()

    R = kernel(graphs)
    D = np.diag(np.diag(R)**-0.5)
    K = D.dot(R).dot(D)

    assert(R.shape == (3, 3))
    for i in range(len(molecules)):
        assert(K[i, i] == pytest.approx(1, 1e-6))
