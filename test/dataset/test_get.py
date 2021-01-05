#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import tempfile
import pytest
from graphdot.dataset import get


def test_get_without_parser():

    with tempfile.TemporaryDirectory() as tempd:
        tempf = os.path.join(tempd, 'robots')
        f = get(
            'https://en.wikipedia.org/robots.txt',
            local_filename=tempf,
            overwrite=False
        )

        assert os.path.exists(f)
        assert os.path.exists(tempf)


def test_get_with_parser():

    with tempfile.TemporaryDirectory() as tempd:
        tempf = os.path.join(tempd, 'robots')
        f = get(
            'https://en.wikipedia.org/robots.txt',
            local_filename=tempf,
            overwrite=False,
            parser=lambda f: open(f).read()
        )

        assert not os.path.exists(f)
        assert os.path.exists(tempf)
        assert isinstance(f, str)
        assert 'wikipedia' in f
