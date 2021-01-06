#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tempfile
import pandas as pd
import pytest
from graphdot.dataset import METLIN_SMRT, AMES


def test_metlin_smrt():
    with tempfile.TemporaryDirectory() as tempd:
        smrt = METLIN_SMRT(local_filename=tempd + '/metlin.csv')
        assert isinstance(smrt, pd.DataFrame)


def test_ames():
    ames = AMES(overwrite=True)
    assert isinstance(ames, dict)
