#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import tempfile
import pytest
from graphdot.dataset.qm7 import QM7


def test_qm7():

    with tempfile.TemporaryDirectory() as tempd:
        tempf = os.path.join(tempd, 'qm7.mat')
        qm7 = QM7(local_filename=tempf, ase=True)

        assert 'columb_matrix' in qm7
        assert 'atomization_energy' in qm7
        assert 'atomic_charge' in qm7
        assert 'xyz' in qm7
        assert 'split' in qm7
        assert 'atoms' in qm7
