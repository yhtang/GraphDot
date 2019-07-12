#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pytest
import networkx as nx
from graphdot import Graph
from graphdot.graphkernel.marginalized import MarginalizedGraphKernel
from graphdot.graphkernel.marginalized.basekernel import Constant
from graphdot.graphkernel.marginalized.basekernel import KroneckerDelta
from graphdot.graphkernel.marginalized.basekernel import SquareExponential
from graphdot.graphkernel.marginalized.basekernel import TensorProduct


def MLGK(G, knode, kedge, q):
    N = len(G.nodes)
    D = np.zeros(N)
    A = np.zeros((N, N))
    Vx = np.zeros(N * N)
    Ex = np.zeros((N * N, N * N))

    for i, n1 in G.nodes.iterrows():
        for j, n2 in G.nodes.iterrows():
            Vx[i*N+j] = knode(n1, n2)

    for (i1, j1), (_, e1) in zip(G.edges['!ij'],
                                 G.edges.drop(['!ij', '!w'], axis=1,
                                              errors='ignore').iterrows()):
        for (i2, j2), (_, e2) in zip(G.edges['!ij'],
                                     G.edges.drop(['!ij', '!w'], axis=1,
                                                  errors='ignore').iterrows()):
            Ex[i1 * N + i2, j1 * N + j2] = kedge(e1, e2)
            Ex[i1 * N + j2, j1 * N + i2] = kedge(e1, e2)
            Ex[j1 * N + j2, i1 * N + i2] = kedge(e1, e2)
            Ex[j1 * N + i2, i1 * N + j2] = kedge(e1, e2)

    if '!w' in G.edges:
        for (i, j), w in zip(G.edges['!ij'], G.edges['!w']):
            D[i] += w
            D[j] += w
            A[i, j] = w
            A[j, i] = w
    else:
        for i, j in G.edges['!ij']:
            D[i] += 1.0
            D[j] += 1.0
            A[i, j] = 1.0
            A[j, i] = 1.0

    Dx = np.kron(D / (1.0 - q), D / (1.0 - q))
    Ax = np.kron(A, A)

    rhs = Dx * np.ones(N * N) * q * q
    linsys = np.diag(Dx / Vx) - Ax * Ex
    solution = np.linalg.solve(linsys, rhs)

    return solution.sum()


def test_mlgk_vanilla():

    g = nx.Graph(title='A')
    g.add_node('a', type=0)
    g.add_node('b', type=0)
    g.add_edge('a', 'b', weight=1.0)
    dfg = Graph.from_networkx(g, weight='weight')

    node_kernel = TensorProduct(type=KroneckerDelta(1.0, 1.0))
    edge_kernel = Constant(1.0)

    q = 0.5
    mlgk = MarginalizedGraphKernel(node_kernel, edge_kernel, q=q)
    dot = mlgk([dfg])

    gold = MLGK(dfg, node_kernel, edge_kernel, q)

    assert(dot.shape == (1, 1))
    assert(np.asscalar(dot) == pytest.approx(gold))


def test_mlgk():

    class Hybrid:
        NONE = np.int32(0)
        SP = np.int32(1)
        SP2 = np.int32(2)
        SP3 = np.int32(3)

    g1 = nx.Graph(title='H2O')
    g1.add_node('O1', hybridization=Hybrid.SP2, charge=np.int32(1))
    g1.add_node('H1', hybridization=Hybrid.SP3, charge=np.int32(-1))
    g1.add_node('H2', hybridization=Hybrid.SP, charge=np.int32(2))
    g1.add_edge('O1', 'H1', order=np.int32(1), length=np.float32(0.5))
    g1.add_edge('O1', 'H2', order=np.int32(2), length=np.float32(1.0))

    g2 = nx.Graph(title='H2')
    g2.add_node('H1', hybridization=Hybrid.SP, charge=np.int32(1))
    g2.add_node('H2', hybridization=Hybrid.SP, charge=np.int32(1))
    g2.add_edge('H1', 'H2', order=np.int32(2), length=np.float32(1.0))

    node_kernel = TensorProduct(hybridization=KroneckerDelta(0.3, 1.0),
                                charge=SquareExponential(1.0))

    edge_kernel = TensorProduct(order=KroneckerDelta(0.3, 1.0),
                                length=SquareExponential(0.05))

    q = 0.1
    mlgk = MarginalizedGraphKernel(node_kernel, edge_kernel, q=q)

    G = [Graph.from_networkx(g) for g in [g1, g2]]
    R = mlgk(G)

    assert(R.shape == (2, 2))
    assert(np.count_nonzero(R - R.T) == 0)
    assert(R[0, 0] == pytest.approx(MLGK(G[0], node_kernel, edge_kernel, q)))
    assert(R[1, 1] == pytest.approx(MLGK(G[1], node_kernel, edge_kernel, q)))


def test_mlgk_weighted():

    class Hybrid:
        NONE = np.int32(0)
        SP = np.int32(1)
        SP2 = np.int32(2)
        SP3 = np.int32(3)

    g1 = nx.Graph(title='H2O')
    g1.add_node('O1', hybridization=Hybrid.SP2, charge=np.int32(1))
    g1.add_node('H1', hybridization=Hybrid.SP3, charge=np.int32(-1))
    g1.add_node('H2', hybridization=Hybrid.SP, charge=np.int32(2))
    g1.add_edge('O1', 'H1', order=np.int32(1), length=np.float32(0.5), w=1.0)
    g1.add_edge('O1', 'H2', order=np.int32(2), length=np.float32(1.0), w=2.0)

    g2 = nx.Graph(title='H2')
    g2.add_node('H1', hybridization=Hybrid.SP, charge=np.int32(1))
    g2.add_node('H2', hybridization=Hybrid.SP, charge=np.int32(1))
    g2.add_edge('H1', 'H2', order=np.int32(2), length=np.float32(1.0), w=3.0)

    node_kernel = TensorProduct(hybridization=KroneckerDelta(0.3, 1.0),
                                charge=SquareExponential(1.0))

    edge_kernel = TensorProduct(order=KroneckerDelta(0.3, 1.0),
                                length=SquareExponential(0.05))
    q = 0.1
    mlgk = MarginalizedGraphKernel(node_kernel, edge_kernel, q=q)

    G = [Graph.from_networkx(g, weight='w') for g in [g1, g2]]

    R = mlgk(G)

    assert(R.shape == (2, 2))
    assert(np.count_nonzero(R - R.T) == 0)
    assert(R[0, 0] == pytest.approx(MLGK(G[0], node_kernel, edge_kernel, q)))
    assert(R[1, 1] == pytest.approx(MLGK(G[1], node_kernel, edge_kernel, q)))
