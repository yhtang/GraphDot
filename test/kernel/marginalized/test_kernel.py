#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pytest
import networkx as nx
from graphdot import Graph
from graphdot.kernel.marginalized import MarginalizedGraphKernel
from graphdot.kernel.marginalized.basekernel import Constant
from graphdot.kernel.marginalized.basekernel import KroneckerDelta
from graphdot.kernel.marginalized.basekernel import SquareExponential
from graphdot.kernel.marginalized.basekernel import TensorProduct


def MLGK(G, knode, kedge, q, q0, nodal=False):
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

    rhs = Dx * np.ones(N * N) * q * q / q0 / q0
    linsys = np.diag(Dx / Vx) - Ax * Ex
    solution = np.linalg.solve(linsys, rhs)

    if nodal is True:
        return solution.reshape(N, -1)
    else:
        return solution.sum()


unlabeled_graph1 = nx.Graph(title='U1')
unlabeled_graph1.add_node(0)
unlabeled_graph1.add_node(1)
unlabeled_graph1.add_node(2)
unlabeled_graph1.add_edge(0, 1)
unlabeled_graph1.add_edge(0, 2)

unlabeled_graph2 = nx.Graph(title='U2')
unlabeled_graph2.add_node(0)
unlabeled_graph2.add_node(1)
unlabeled_graph2.add_node(2)
unlabeled_graph2.add_edge(0, 1)
unlabeled_graph2.add_edge(0, 2)
unlabeled_graph2.add_edge(1, 2)


class Hybrid:
    NONE = 0
    SP = 1
    SP2 = 2
    SP3 = 3


labeled_graph1 = nx.Graph(title='H2O')
labeled_graph1.add_node('O1', hybridization=Hybrid.SP2, charge=1)
labeled_graph1.add_node('H1', hybridization=Hybrid.SP3, charge=-1)
labeled_graph1.add_node('H2', hybridization=Hybrid.SP, charge=2)
labeled_graph1.add_edge('O1', 'H1', order=1, length=0.5)
labeled_graph1.add_edge('O1', 'H2', order=2, length=1.0)

labeled_graph2 = nx.Graph(title='H2')
labeled_graph2.add_node('H1', hybridization=Hybrid.SP, charge=1)
labeled_graph2.add_node('H2', hybridization=Hybrid.SP, charge=1)
labeled_graph2.add_edge('H1', 'H2', order=2, length=1.0)

weighted_graph1 = nx.Graph(title='H2O')
weighted_graph1.add_node('O1', hybridization=Hybrid.SP2, charge=1)
weighted_graph1.add_node('H1', hybridization=Hybrid.SP3, charge=-1)
weighted_graph1.add_node('H2', hybridization=Hybrid.SP, charge=2)
weighted_graph1.add_edge('O1', 'H1', order=1, length=0.5, w=1.0)
weighted_graph1.add_edge('O1', 'H2', order=2, length=1.0, w=2.0)

weighted_graph2 = nx.Graph(title='H2')
weighted_graph2.add_node('H1', hybridization=Hybrid.SP, charge=1)
weighted_graph2.add_node('H2', hybridization=Hybrid.SP, charge=1)
weighted_graph2.add_edge('H1', 'H2', order=2, length=1.0, w=3.0)


def test_mlgk_typecheck():
    node_kernel = Constant(1.0)
    edge_kernel = Constant(1.0)
    mlgk = MarginalizedGraphKernel(node_kernel, edge_kernel, q=0.5)
    G = [Graph.from_networkx(unlabeled_graph1),
         Graph.from_networkx(labeled_graph1),
         Graph.from_networkx(weighted_graph1, weight='w')]

    with pytest.raises(TypeError):
        mlgk([G[0], G[1]])
    with pytest.raises(TypeError):
        mlgk([G[0], G[2]])
    with pytest.raises(TypeError):
        mlgk([G[1], G[2]])
    with pytest.raises(TypeError):
        mlgk([G[1], G[0]])
    with pytest.raises(TypeError):
        mlgk([G[2], G[0]])
    with pytest.raises(TypeError):
        mlgk([G[2], G[1]])


@pytest.mark.parametrize('caseset', {
    'unlabeled': {
        'graphs': [Graph.from_networkx(unlabeled_graph1),
                   Graph.from_networkx(unlabeled_graph2)],
        'knode': Constant(1.0),
        'kedge': Constant(1.0),
        'q': [0.01, 0.05, 0.1, 0.5]
    },
    'labeled': {
        'graphs': [Graph.from_networkx(labeled_graph1),
                   Graph.from_networkx(labeled_graph2)],
        'knode': TensorProduct(hybridization=KroneckerDelta(0.3, 1.0),
                               charge=SquareExponential(1.0)),
        'kedge': TensorProduct(order=KroneckerDelta(0.3, 1.0),
                               length=SquareExponential(0.05)),
        'q': [0.01, 0.05, 0.1, 0.5]
    },
    'weighted': {
        'graphs': [Graph.from_networkx(weighted_graph1, weight='w'),
                   Graph.from_networkx(weighted_graph2, weight='w')],
        'knode': TensorProduct(hybridization=KroneckerDelta(0.3, 1.0),
                               charge=SquareExponential(1.0)),
        'kedge': TensorProduct(order=KroneckerDelta(0.3, 1.0),
                               length=SquareExponential(0.05)),
        'q': [0.01, 0.05, 0.1, 0.5]
    },
}.items())
def test_mlgk(caseset):

    casetitle, case = caseset
    G = case['graphs']
    knode = case['knode']
    kedge = case['kedge']
    for q in case['q']:

        mlgk = MarginalizedGraphKernel(knode, kedge, q=q)

        R = mlgk(G)
        d = np.diag(R)**-0.5
        K = np.diag(d).dot(R).dot(np.diag(d))

        assert(R.shape == (len(G), len(G)))
        assert(np.count_nonzero(R - R.T) == 0)
        assert(R[0, 0] == pytest.approx(MLGK(G[0], knode, kedge, q, q), 1e-5))
        assert(R[1, 1] == pytest.approx(MLGK(G[1], knode, kedge, q, q), 1e-5))
        assert(K[0, 0] == pytest.approx(1, 1e-7))
        assert(K[1, 1] == pytest.approx(1, 1e-7))

        for x, y in zip(mlgk(G[:1], G).ravel(), R[:1, :].ravel()):
            assert(x == pytest.approx(y, 1e-6))
        for x, y in zip(mlgk(G[1:], G).ravel(), R[1:, :].ravel()):
            assert(x == pytest.approx(y, 1e-6))
        for x, y in zip(mlgk(G, G[:1]).ravel(), R[:, :1].ravel()):
            assert(x == pytest.approx(y, 1e-6))
        for x, y in zip(mlgk(G, G[1:],).ravel(), R[:, 1:].ravel()):
            assert(x == pytest.approx(y, 1e-6))

        R_nodal = mlgk(G, nodal=True)
        d_nodal = np.diag(R_nodal)**-0.5
        K_nodal = np.diag(d_nodal).dot(R_nodal).dot(np.diag(d_nodal))

        n = np.array([len(g.nodes) for g in G])
        N = np.cumsum(n)
        assert(R_nodal.shape == (N[-1], N[-1]))
        assert(np.count_nonzero(R_nodal - R_nodal.T) == 0)
        for k, (i, j) in enumerate(zip(N-n, N)):
            gnd = MLGK(G[k], knode, kedge, q, q, nodal=True).ravel()
            sub = R_nodal[i:j, :][:, i:j].ravel()
            for r1, r2 in zip(sub, gnd):
                assert(r1 == pytest.approx(r2, 1e-5))
        for i in range(N[-1]):
            assert(K_nodal[i, i] == pytest.approx(1, 1e-7))

        mlgk = MarginalizedGraphKernel(knode, kedge, q=q,
                                       p=lambda node: 2.0)
        R = mlgk(G)
        gnd_R00 = MLGK(G[0], knode, kedge, q, q) * 2.0**2
        gnd_R11 = MLGK(G[1], knode, kedge, q, q) * 2.0**2
        assert(R[0, 0] == pytest.approx(gnd_R00, 1e-5))
        assert(R[1, 1] == pytest.approx(gnd_R11, 1e-5))

        for i1, j1, g1 in zip(N-n, N, G):
            for i2, j2, g2 in zip(N-n, N, G):
                gnd = R_nodal[i1:j1, :][:, i2:j2] * 2.0**2
                sub = mlgk([g1], [g2], nodal=True)
                for r1, r2 in zip(sub, gnd):
                    assert(r1 == pytest.approx(r2, 1e-5))

# def test_mlgk_large():
#     g = nx.Graph()
#     n = 24
#     for i, row in enumerate(np.random.randint(0, 2, (n, n))):
#         g.add_node(i, type=0)
#         for j, pred in enumerate(row[:i]):
#             if pred:
#                 g.add_edge(i, j, weight=1)
#
#     dfg = Graph.from_networkx(g, weight='weight')
#
#     q = 0.5
#     node_kernel = TensorProduct(type=KroneckerDelta(1.0, 1.0))
#     edge_kernel = Constant(1.0)
#     mlgk = MarginalizedGraphKernel(node_kernel, edge_kernel, q=q)
#
#     dot = mlgk([dfg])
#     gold = MLGK(dfg, node_kernel, edge_kernel, q)
#
#     assert(dot.shape == (1, 1))
#     assert(dot.item() == pytest.approx(gold))

    #
    #
    # q = 0.1
    # node_kernel = TensorProduct(hybridization=KroneckerDelta(0.3, 1.0),
    #                             charge=SquareExponential(1.0))
    #
    # edge_kernel = TensorProduct(order=KroneckerDelta(0.3, 1.0),
    #                             length=SquareExponential(0.05))
    # mlgk = MarginalizedGraphKernel(node_kernel, edge_kernel, q=q)
    #
    # R = mlgk(G)
    #
    # assert(R.shape == (2, 2))
    # assert(np.count_nonzero(R - R.T) == 0)
    # assert(R[0, 0] == pytest.approx(MLGK(G[0], node_kernel, edge_kernel, q)))
    # assert(R[1, 1] == pytest.approx(MLGK(G[1], node_kernel, edge_kernel, q)))