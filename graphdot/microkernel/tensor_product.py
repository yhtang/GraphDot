#!/usr/bin/env python
# -*- coding: utf-8 -*-
from .composite import Composite


def TensorProduct(**kw_kernels):
    r"""Alias of `Composite('*', **kw_kernels)`.
    :math:`k_\otimes(X, Y) = \prod_{a \in \mathrm{features}} k_a(X_a, Y_a)`
    """
    return Composite('*', **kw_kernels)
