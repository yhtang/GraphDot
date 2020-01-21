#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest
from numpy import nan
from ase.build import molecule
from graphdot.experimental.metric.m3 import M3


def test_m3():
    atoms1 = molecule('CH4')
    atoms2 = molecule('CH4')
    atoms2.positions[0, 0] += 0.03

    m3 = M3()

    eps = 1e-7
    assert(m3(atoms1, atoms1) == pytest.approx(0, abs=eps))
    assert(m3(atoms2, atoms2) == pytest.approx(0, abs=eps))
    assert(m3(atoms1, atoms2) > eps)
    print('m3(atoms1, atoms2)', m3(atoms1, atoms2))


def test_m3_charge():
    atoms1 = molecule('CH4')
    atoms2 = molecule('CH4')
    atoms1.set_initial_charges([0, 0, 0, 0, 0])
    atoms2.set_initial_charges([0, 0, 0, 0, 0.1])

    m3 = M3(use_charge=True)

    eps = 1e-7
    assert(m3(atoms1, atoms1) == pytest.approx(0, abs=eps))
    assert(m3(atoms2, atoms2) == pytest.approx(0, abs=eps))
    assert(m3(atoms1, atoms2) > eps)
    print('m3(atoms1, atoms2)', m3(atoms1, atoms2))
