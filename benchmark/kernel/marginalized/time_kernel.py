#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest
import numpy as np
import pycuda.autoinit
from networkx.generators.random_graphs import newman_watts_strogatz_graph
from graphdot import Graph
from graphdot.kernel.marginalized import MarginalizedGraphKernel
from graphdot.kernel.marginalized.basekernel import Constant
from graphdot.kernel.marginalized.basekernel import TensorProduct
from graphdot.kernel.marginalized.basekernel import KroneckerDelta


def make_graphs(batch, size):
    '''make random graphs'''
    nxgraphs = []
    np.random.seed(0)
    for _ in range(batch):
        # make topology
        g = newman_watts_strogatz_graph(size, k=5, p=0.05, seed=0)
        # assign vertex label
        for i in range(size):
            g.nodes[i]['label'] = np.random.randint(0, 9)
        # assign edge weight and label
        for ij in g.edges:
            g.edges[ij]['label'] = np.random.randint(0, 9)
            g.edges[ij]['weight'] = np.random.randint(1, 5)
        nxgraphs.append(g)
    return nxgraphs


def test_marginalized_graph_kernel_init(benchmark):

    knode = Constant(1)
    kedge = Constant(1)

    def fun():
        return MarginalizedGraphKernel(knode, kedge)

    benchmark.pedantic(fun, iterations=5, rounds=5, warmup_rounds=1)


@pytest.mark.parametrize("batch", [1, 16, 128])
def test_marginalized_graph_kernel_1st_launch(benchmark, batch):

    graphs = [Graph.from_networkx(g, weight='weight')
              for g in make_graphs(batch, 48)]

    knode = TensorProduct(label=KroneckerDelta(0.5))
    kedge = TensorProduct(label=KroneckerDelta(0.5))

    def fun():
        kernel = MarginalizedGraphKernel(knode, kedge)
        kernel(graphs, nodal=False)

    benchmark.pedantic(fun, iterations=3, rounds=3, warmup_rounds=0)


@pytest.mark.parametrize("batch", [1, 16, 128])
def test_marginalized_graph_kernel_2nd_launch(benchmark, batch):

    graphs = [Graph.from_networkx(g, weight='weight')
              for g in make_graphs(batch, 48)]

    knode = TensorProduct(label=KroneckerDelta(0.5))
    kedge = TensorProduct(label=KroneckerDelta(0.5))
    kernel = MarginalizedGraphKernel(knode, kedge)

    def fun():
        kernel(graphs, nodal=False)

    benchmark.pedantic(fun, iterations=3, rounds=3, warmup_rounds=0)
