#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest
from numpy import nan
from ase.build import molecule
from graphdot.experimental.metric.m3 import M3


def test_m3():
    atoms1 = molecule('CH4')
    atoms2 = molecule('CH4')
    atoms2.positions[0, 0] += 0.001

    m3 = M3()

    eps = 1e-7
    assert(m3(atoms1, atoms1) == pytest.approx(0, abs=eps))
    assert(m3(atoms2, atoms2) == pytest.approx(0, abs=eps))
    assert(m3(atoms1, atoms2) > eps)


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


def test_m3_charge_optional():
    atoms1 = molecule('CH4')
    atoms2 = molecule('CH4')
    atoms3 = molecule('CH4')
    atoms4 = molecule('CH4')
    atoms1.set_initial_charges([0, 1, 0, -1, 0])
    atoms2.set_initial_charges([0, 1, 0, nan, 0])
    atoms3.set_initial_charges([0, 1, 0, -0.99, 0])
    atoms4.set_initial_charges([0, 1, 0, 0, 0])

    m3 = M3(use_charge=True)

    eps = 1e-7
    d11 = m3(atoms1, atoms1)
    d12 = m3(atoms1, atoms2)
    d13 = m3(atoms1, atoms3)
    d14 = m3(atoms1, atoms4)
    d44 = m3(atoms4, atoms4)
    assert(d11 <= d12)
    assert(d12 < d13)
    assert(d13 < d14)
    assert(d44 == pytest.approx(0, abs=eps))
