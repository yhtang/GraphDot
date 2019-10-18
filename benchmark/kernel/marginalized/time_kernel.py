#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest
from graphdot.kernel.marginalized import MarginalizedGraphKernel
from graphdot.kernel.marginalized.basekernel import Constant


def test_marginalized_graph_kernel_init(benchmark):

    knode = Constant(1)
    kedge = Constant(1)

    def fun():
        return MarginalizedGraphKernel(knode, kedge)

    benchmark.pedantic(fun, iterations=5, rounds=5, warmup_rounds=1)
