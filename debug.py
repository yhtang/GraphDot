#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import scipy.sparse as sp
import networkx as nx
from graphdot import Graph
from graphdot.kernel.marginalized import MarginalizedGraphKernel
from graphdot.kernel.basekernel import Constant
from graphdot.kernel.basekernel import KroneckerDelta
from graphdot.kernel.basekernel import SquareExponential
from graphdot.kernel.basekernel import TensorProduct

np.set_printoptions(linewidth=999, precision=4, suppress=True)


class Hybrid:
    NONE = 0
    SP = 1
    SP2 = 2
    SP3 = 3


def MLGK(G, knode, kedge, q, q0):
    N = len(G.nodes)
    D = np.zeros(N)
    A = np.zeros((N, N))
    Vx = np.zeros(N * N)
    Ex = np.zeros((N * N, N * N))

    for i, n1 in G.nodes.iterrows():
        for j, n2 in G.nodes.iterrows():
            Vx[i*N+j] = knode(n1, n2)

    for i1, j1, e1 in zip(G.edges['!i'], G.edges['!j'], G.edges.rows()):
        for i2, j2, e2 in zip(G.edges['!i'], G.edges['!j'], G.edges.rows()):
            Ex[i1 * N + i2, j1 * N + j2] = kedge(e1, e2)
            Ex[i1 * N + j2, j1 * N + i2] = kedge(e1, e2)
            Ex[j1 * N + j2, i1 * N + i2] = kedge(e1, e2)
            Ex[j1 * N + i2, i1 * N + j2] = kedge(e1, e2)

    if '!w' in G.edges:
        for i, j, w in zip(G.edges['!i'], G.edges['!j'], G.edges['!w']):
            D[i] += w
            D[j] += w
            A[i, j] = w
            A[j, i] = w
    else:
        for i, j in zip(G.edges['!i'], G.edges['!j']):
            D[i] += 1.0
            D[j] += 1.0
            A[i, j] = 1.0
            A[j, i] = 1.0

    Dx = np.kron(D / (1.0 - q), D / (1.0 - q))
    Ax = np.kron(A, A)

    rhs = Dx * np.ones(N * N) * q * q / q0 / q0
    linsys = np.diag(Dx / Vx) - Ax * Ex
    solution, _ = sp.linalg.cg(linsys, rhs, atol=1e-7,
                               M=sp.diags([Vx / Dx], [0]))
    kappa = solution.sum()

    dK = np.zeros((N * N, N * N))
    eps = 1e-3
    Px = Ax / Dx[:, None]
    Z = np.diag(1 / Vx) - Px * Ex
    rhs4z = np.ones(N * N) * q * q / q0 / q0
    for i in range(N * N):
        for j in range(N * N):
            Zr = Z.copy()
            Zr[i, j] += eps
            right = np.linalg.solve(Zr, rhs4z)
            Zl = Z.copy()
            Zl[i, j] -= eps
            left = np.linalg.solve(Zl, rhs4z)
            dK[i, j] = (right.sum() - left.sum()) / (2 * eps)

    DVx = np.zeros((N * N, len(knode.theta)))

    for i, n1 in G.nodes.iterrows():
        for j, n2 in G.nodes.iterrows():
            f, jac = knode(n1, n2, jac=True)
            DVx[i*N+j, :] = jac

    return kappa, dK, np.sum(dK.diagonal()[:, None] * DVx, axis=0)


weighted_graph1 = nx.Graph(title='H2O')
weighted_graph1.add_node('O1', hybridization=Hybrid.SP2, charge=1)
weighted_graph1.add_node('H1', hybridization=Hybrid.SP3, charge=-1)
weighted_graph1.add_node('H2', hybridization=Hybrid.SP, charge=2)
weighted_graph1.add_edge('O1', 'H1', order=1, length=0.5, w=1.0)
weighted_graph1.add_edge('O1', 'H2', order=2, length=1.0, w=2.0)

graph = Graph.from_networkx(weighted_graph1, weight='w')
knode = TensorProduct(hybridization=KroneckerDelta(0.3),
                      charge=SquareExponential(1.0))
kedge = TensorProduct(order=KroneckerDelta(0.3),
                      length=SquareExponential(1.0))

mlgk = MarginalizedGraphKernel(knode, kedge, q=0.1)

# print(graph)

R, dR = mlgk([graph], eval_gradient=True, nodal=False)

print(mlgk.hyperparameters)

print(R)

print(dR)

# print(R.sum())

# print(-np.outer(dR[:, :, 0].ravel(), R.ravel()))

# print(mlgk.backend.source)

K, dKdZ, dKdV = MLGK(graph, knode, kedge, q=0.1, q0=0.1)

print(K)

print(dKdZ)

print(dKdV)

eps = 1e-2
for i in range(len(mlgk.theta)):
    theta = mlgk.theta

    t = np.exp(theta)
    t[i] += eps
    mlgk.theta = np.log(t)
    Rr = mlgk([graph])

    t = np.exp(theta)
    t[i] -= eps
    mlgk.theta = np.log(t)
    Rl = mlgk([graph])

    mlgk.theta = theta

    print((Rr - Rl)/(2 * eps))
