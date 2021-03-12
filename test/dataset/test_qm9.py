#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import tempfile
import pytest
from graphdot.dataset.qm9 import QM9


def test_qm9():

    with tempfile.TemporaryDirectory() as tempd:
        tempf = os.path.join(tempd, 'qm7.mat')
        qm9 = QM9(local_filename=tempf, ase=True)

        assert len(qm9) > 133000
        assert 'tag' in qm9
        assert 'zpve' in qm9
        assert 'atoms' in qm9
