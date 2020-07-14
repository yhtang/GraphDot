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

    R_nodal = kernel(graphs, nodal=True)
    D_nodal = np.diag(np.diag(R_nodal)**-0.5)
    K_nodal = D_nodal.dot(R_nodal).dot(D_nodal)

    natoms = np.sum([len(m) for m in molecules])
    assert(R_nodal.shape == (natoms, natoms))
    for i in range(natoms):
        assert(K_nodal[i, i] == pytest.approx(1, 1e-6))


def test_molecular_kernel_custom_pstart():
    molecules = [molecule('H2'), molecule('O2'), molecule('CH4')]

    graphs = [Graph.from_ase(m) for m in molecules]

    kernel_nocarbon = Tang2019MolecularKernel(
        starting_probability=(
            lambda ns: np.where(ns.element == 6, 0.0, 1.0),
            'n.element == 6 ? 0.f : 1.f'
        )
    )

    R_nocarbon_nodal = kernel_nocarbon(graphs, nodal=True)
    k = 0
    for i, m in enumerate(molecules):
        for j, a in enumerate(m):
            if a.symbol == 'C':
                assert(R_nocarbon_nodal[k, :].sum() == 0)
                assert(R_nocarbon_nodal[:, k].sum() == 0)
            k += 1
