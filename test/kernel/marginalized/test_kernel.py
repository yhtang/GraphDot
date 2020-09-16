#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import scipy.sparse as sp
import pytest
import networkx as nx
from ase.build import molecule
from graphdot import Graph
from graphdot.kernel.marginalized import MarginalizedGraphKernel
from graphdot.microkernel import (
    Constant,
    KroneckerDelta,
    SquareExponential,
    TensorProduct,
    Additive,
    Convolution
)


def MLGK(G, knode, kedge, q, q0, nodal=False):
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
            if i2 != j2:
                Ex[i1 * N + j2, j1 * N + i2] = kedge(e1, e2)
            if i1 != j1:
                Ex[j1 * N + j2, i1 * N + i2] = kedge(e1, e2)
                if i2 != j2:
                    Ex[j1 * N + i2, i1 * N + j2] = kedge(e1, e2)

    if '!w' in G.edges:
        for i, j, w in zip(G.edges['!i'], G.edges['!j'], G.edges['!w']):
            D[i] += w
            if i != j:
                D[j] += w
            A[i, j] = w
            A[j, i] = w
    else:
        for i, j in zip(G.edges['!i'], G.edges['!j']):
            D[i] += 1.0
            if i != j:
                D[j] += 1.0
            A[i, j] = 1.0
            A[j, i] = 1.0

    Dx = np.kron(D / (1.0 - q), D / (1.0 - q))
    Ax = np.kron(A, A)
    linsys = np.diag(Dx / Vx) - Ax * Ex

    qx = np.ones(N * N) * q * q / q0 / q0
    px = np.ones(N * N)

    solution, _ = sp.linalg.cg(linsys, Dx * qx, atol=1e-7)

    if nodal is True:
        return solution.reshape(N, -1)
    else:
        return px.dot(solution)


class Hybrid:
    NONE = 0
    SP = 1
    SP2 = 2
    SP3 = 3


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

labeled_graph1 = nx.Graph(title='H2O')
labeled_graph1.add_node('O1', hybridization=Hybrid.SP2, charge=1)
labeled_graph1.add_node('H1', hybridization=Hybrid.SP3, charge=-1)
labeled_graph1.add_node('H2', hybridization=Hybrid.SP, charge=2)
labeled_graph1.add_edge('O1', 'H1', order=1, length=0.5)
labeled_graph1.add_edge('O1', 'H2', order=2, length=1.0)

labeled_graph2 = nx.Graph(title='H2')
labeled_graph2.add_node('H1', hybridization=Hybrid.SP, charge=1.0)
labeled_graph2.add_node('H2', hybridization=Hybrid.SP, charge=-1.0)
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

vario_graph1 = nx.Graph(title='H2O')
vario_graph1.add_node('O1', rings=(5, 6))
vario_graph1.add_node('H1', rings=(3,))
vario_graph1.add_node('H2', rings=(2, 3, 4))
vario_graph1.add_edge('O1', 'H1', spectrum=(3, 4), w=1.0)
vario_graph1.add_edge('O1', 'H2', spectrum=(3, 5), w=2.0)

vario_graph2 = nx.Graph(title='H2')
vario_graph2.add_node('H1', rings=(3, 4))
vario_graph2.add_node('H2', rings=(3, ))
vario_graph2.add_edge('H1', 'H2', spectrum=(2, 4), w=3.0)

case_dict = {
    'unlabeled': {
        'graphs': Graph.unify_datatype([
            Graph.from_networkx(unlabeled_graph1),
            Graph.from_networkx(unlabeled_graph2)
        ]),
        'knode': Constant(1.0),
        'kedge': Constant(1.0),
        'q': [0.01, 0.05, 0.1, 0.5]
    },
    'labeled': {
        'graphs': Graph.unify_datatype([
            Graph.from_networkx(labeled_graph1),
            Graph.from_networkx(labeled_graph2)
        ]),
        'knode': TensorProduct(hybridization=KroneckerDelta(0.3),
                               charge=SquareExponential(1.) + 0.01).normalized,
        'kedge': Additive(order=KroneckerDelta(0.3),
                          length=SquareExponential(0.05)).normalized,
        'q': [0.01, 0.05, 0.1, 0.5]
    },
    'weighted': {
        'graphs': Graph.unify_datatype([
            Graph.from_networkx(weighted_graph1, weight='w'),
            Graph.from_networkx(weighted_graph2, weight='w')
        ]),
        'knode': Additive(hybridization=KroneckerDelta(0.3),
                          charge=SquareExponential(1.0)).normalized,
        'kedge': TensorProduct(order=KroneckerDelta(0.3),
                               length=SquareExponential(0.05)),
        'q': [0.01, 0.05, 0.1, 0.5]
    },
    'vario-features': {
        'graphs': Graph.unify_datatype([
            Graph.from_networkx(vario_graph1, weight='w'),
            Graph.from_networkx(vario_graph2, weight='w')
        ]),
        'knode': TensorProduct(rings=Convolution(KroneckerDelta(0.3))),
        'kedge': TensorProduct(spectrum=Convolution(SquareExponential(1.0))),
        'q': [0.01, 0.05, 0.1, 0.5]
    },
}


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


@pytest.mark.parametrize('caseitem', case_dict.items())
def test_mlgk_self_similarity(caseitem):
    '''overall similarities within X'''

    _, case = caseitem

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
        assert(K[0, 0] == pytest.approx(1, 2e-7))
        assert(K[1, 1] == pytest.approx(1, 2e-7))


@pytest.mark.parametrize('caseitem', case_dict.items())
def test_mlgk_cross_similarity(caseitem):
    '''similarities across X and Y'''

    _, case = caseitem

    G = case['graphs']
    knode = case['knode']
    kedge = case['kedge']
    for q in case['q']:

        mlgk = MarginalizedGraphKernel(knode, kedge, q=q)
        R = mlgk(G)

        for x, y in zip(mlgk(G[:1], G).ravel(), R[:1, :].ravel()):
            assert(x == pytest.approx(y, 1e-6))
        for x, y in zip(mlgk(G[1:], G).ravel(), R[1:, :].ravel()):
            assert(x == pytest.approx(y, 1e-6))
        for x, y in zip(mlgk(G, G[:1]).ravel(), R[:, :1].ravel()):
            assert(x == pytest.approx(y, 1e-6))
        for x, y in zip(mlgk(G, G[1:],).ravel(), R[:, 1:].ravel()):
            assert(x == pytest.approx(y, 1e-6))


@pytest.mark.parametrize('caseitem', case_dict.items())
def test_mlgk_gradient(caseitem):
    '''derivative w.r.t. hyperparameters'''

    _, case = caseitem

    G = case['graphs']
    knode = case['knode']
    kedge = case['kedge']
    for q in case['q']:

        mlgk = MarginalizedGraphKernel(knode, kedge, q=q)

        np.set_printoptions(precision=4, linewidth=999, suppress=True)

        R, dR = mlgk(G, nodal=False, eval_gradient=True)

        assert(len(dR.shape) == 3)
        assert(R.shape[0] == dR.shape[0])
        assert(R.shape[1] == dR.shape[1])
        assert(dR.shape[2] >= 1)

        for i in range(len(mlgk.theta)):

            theta = mlgk.theta

            eps = 1e-3

            t = np.copy(theta)
            t[i] += eps
            mlgk.theta = t
            Rr = mlgk(G)

            t = np.copy(theta)
            t[i] -= eps
            mlgk.theta = t
            Rl = mlgk(G)

            mlgk.theta = theta

            dR_dLogt = (Rr - Rl) / (2 * eps)
            dLogt_dt = 1 / np.exp(theta)[i]
            dR_dt = dR_dLogt * dLogt_dt

            for a, b in zip(dR[:, :, i].ravel(), dR_dt.ravel()):
                assert(a == pytest.approx(b, rel=0.05, abs=0.05))


@pytest.mark.parametrize('caseitem', case_dict.items())
def test_mlgk_diag(caseitem):
    '''diagonal similarities'''

    _, case = caseitem

    G = case['graphs']
    knode = case['knode']
    kedge = case['kedge']
    for q in case['q']:

        mlgk = MarginalizedGraphKernel(knode, kedge, q=q)
        R = mlgk(G)

        D = mlgk.diag(G)
        assert(len(D) == 2)
        assert(D[0] == pytest.approx(R[0, 0], 1e-7))
        assert(D[1] == pytest.approx(R[1, 1], 1e-7))

        '''nodal diags'''
        R_nodal = mlgk(G, nodal=True)
        d_nodal = np.diag(R_nodal)**-0.5
        K_nodal = np.diag(d_nodal).dot(R_nodal).dot(np.diag(d_nodal))

        '''check submatrices'''
        n = np.array([len(g.nodes) for g in G])
        N = np.cumsum(n)
        start = N - n
        end = N
        assert(R_nodal.shape == (N[-1], N[-1]))
        assert(np.count_nonzero(R_nodal - R_nodal.T) == 0)
        for k, (i, j) in enumerate(zip(N - n, N)):
            gnd = MLGK(G[k], knode, kedge, q, q, nodal=True).ravel()
            sub = R_nodal[i:j, :][:, i:j].ravel()
            for r1, r2 in zip(sub, gnd):
                assert(r1 == pytest.approx(r2, 1e-5))
        for i in range(N[-1]):
            assert(K_nodal[i, i] == pytest.approx(1, 2e-7))

        '''check block-diags'''
        D_nodal = mlgk.diag(G, nodal=True)
        assert(len(D_nodal) == N[-1])
        for k in range(2):
            i = start[k]
            j = end[k]
            sub = D_nodal[i:j]
            gnd = np.diag(R_nodal[i:j, :][:, i:j])
            for r1, r2 in zip(sub, gnd):
                assert(r1 == pytest.approx(r2, 1e-7))


@pytest.mark.parametrize('caseitem', case_dict.items())
def test_mlgk_diag_gradient(caseitem):
    '''derivative w.r.t. hyperparameters'''

    _, case = caseitem

    G = case['graphs']
    knode = case['knode']
    kedge = case['kedge']
    for q in case['q']:

        mlgk = MarginalizedGraphKernel(knode, kedge, q=q)

        R, dR = mlgk.diag(G, nodal=False, eval_gradient=True)

        assert(len(dR.shape) == 2)
        assert(R.shape[0] == dR.shape[0])
        assert(dR.shape[1] >= 1)

        for i in range(len(mlgk.theta)):

            theta = mlgk.theta

            eps = 1e-3

            t = np.copy(theta)
            t[i] += eps
            mlgk.theta = t
            Rr = mlgk.diag(G, nodal=False, eval_gradient=False)

            t = np.copy(theta)
            t[i] -= eps
            mlgk.theta = t
            Rl = mlgk.diag(G, nodal=False, eval_gradient=False)

            mlgk.theta = theta

            dR_dLogt = (Rr - Rl) / (2 * eps)
            dLogt_dt = 1 / np.exp(theta)[i]
            dR_dt = dR_dLogt * dLogt_dt

            for a, b in zip(dR[:, i].ravel(), dR_dt.ravel()):
                assert(a == pytest.approx(b, rel=0.05, abs=0.05))


@pytest.mark.parametrize('caseitem', case_dict.items())
def test_mlgk_lmin(caseitem):
    '''exclude first step'''

    _, case = caseitem

    G = case['graphs']
    knode = case['knode']
    kedge = case['kedge']
    for q in case['q']:

        mlgk = MarginalizedGraphKernel(knode, kedge, q=q)

        g = G[0]
        R0 = mlgk([g], nodal=True, lmin=0)
        R1 = mlgk([g], nodal=True, lmin=1)
        for i, n1 in g.nodes.iterrows():
            for j, n2 in g.nodes.iterrows():
                assert(R0[i, j] == pytest.approx(R1[i, j] + knode(n1, n2),
                                                 abs=1e-6))


@pytest.mark.parametrize('caseitem', case_dict.items())
def test_mlgk_starting_probability(caseitem):
    '''custom starting probability'''

    _, case = caseitem

    G = case['graphs']
    knode = case['knode']
    kedge = case['kedge']
    for q in case['q']:

        mlgk = MarginalizedGraphKernel(knode, kedge, q=q, p=2.0)
        R = mlgk(G)
        R_nodal = mlgk(G, nodal=True)
        gnd_R00 = MLGK(G[0], knode, kedge, q, q) * 2.0**2
        gnd_R11 = MLGK(G[1], knode, kedge, q, q) * 2.0**2
        assert(R[0, 0] == pytest.approx(gnd_R00, 1e-5))
        assert(R[1, 1] == pytest.approx(gnd_R11, 1e-5))

        n = np.array([len(g.nodes) for g in G])
        N = np.cumsum(n)
        start = N - n
        end = N
        for i1, j1, g1 in zip(start, end, G):
            for i2, j2, g2 in zip(start, end, G):
                gnd = R_nodal[i1:j1, :][:, i2:j2]
                sub = mlgk([g1], [g2], nodal=True)
                for r1, r2 in zip(sub, gnd):
                    assert(r1 == pytest.approx(r2, 1e-5))


def test_mlgk_large():
    g = nx.Graph()
    n = 24
    for i, row in enumerate(np.random.randint(0, 2, (n, n))):
        g.add_node(i, type=0)
        for j, pred in enumerate(row[:i]):
            if pred:
                g.add_edge(i, j, weight=1)

    dfg = Graph.from_networkx(g, weight='weight')

    q = 0.5
    node_kernel = TensorProduct(type=KroneckerDelta(1.0))
    edge_kernel = Constant(1.0)
    mlgk = MarginalizedGraphKernel(node_kernel, edge_kernel, q=q)

    dot = mlgk([dfg])
    gold = MLGK(dfg, node_kernel, edge_kernel, q, q)

    assert(dot.shape == (1, 1))
    assert(dot.item() == pytest.approx(gold))


def test_mlgk_dtype():
    g = nx.Graph()
    n = 8
    for i, row in enumerate(np.random.randint(0, 2, (n, n))):
        g.add_node(i, type=0)
        for j, pred in enumerate(row[:i]):
            if pred:
                g.add_edge(i, j, weight=1)

    dfg = Graph.from_networkx(g, weight='weight')

    q = 0.5
    node_kernel = TensorProduct(type=KroneckerDelta(1.0))
    edge_kernel = Constant(1.0)

    for dtype in [np.float, np.float32, np.float64]:
        mlgk = MarginalizedGraphKernel(
            node_kernel,
            edge_kernel,
            q=q,
            dtype=dtype
        )

        assert(mlgk([dfg]).dtype == dtype)
        assert(mlgk.diag([dfg]).dtype == dtype)


def test_mlgk_on_permuted_graph():
    g = Graph.from_ase(molecule('C6H6'))
    for _ in range(10):
        h = g.permute(np.random.permutation(len(g.nodes)))
        kernel = MarginalizedGraphKernel(
            TensorProduct(
                element=KroneckerDelta(0.5)
            ),
            TensorProduct(
                length=SquareExponential(0.1)
            )
        )
        assert(kernel([g], [h]).item() == pytest.approx(kernel([g]).item()))


def test_mlgk_self_loops():

    kedge = Constant(1.0)
    knode = Constant(1.0)
    q = 0.1
    mlgk = MarginalizedGraphKernel(knode, kedge, q=q)

    np.random.seed(2)
    for i in range(10):
        n = np.random.randint(4, 20)
        A = np.random.randn(n, n)
        A = A + A.T

        G = [Graph.from_networkx(nx.from_numpy_array(A), weight='weight')]

        K = mlgk(G).item()
        K0 = MLGK(G[0], knode, kedge, q, q, nodal=False)

        assert(K == pytest.approx(K0, 5e-4))


def test_mlgk_fixed_hyperparameters():

    g = nx.Graph()
    g.add_node(0, feature=0)
    g.add_node(1, feature=1)
    g.add_node(2, feature=0)
    g.add_edge(0, 1, attribute=1.0)
    g.add_edge(0, 2, attribute=2.0)

    G = [Graph.from_networkx(g)]
    knodeV = TensorProduct(feature=KroneckerDelta(0.5))
    knodeF = TensorProduct(feature=KroneckerDelta(0.5, h_bounds='fixed'))
    kedgeV = TensorProduct(attribute=SquareExponential(1.0))
    kedgeF = TensorProduct(
        attribute=SquareExponential(1.0, length_scale_bounds='fixed')
    )

    kernelVV = MarginalizedGraphKernel(knodeV, kedgeV)
    kernelVF = MarginalizedGraphKernel(knodeV, kedgeF)
    kernelFV = MarginalizedGraphKernel(knodeF, kedgeV)
    kernelFF = MarginalizedGraphKernel(knodeF, kedgeF)
    assert(len(kernelVV.theta) == len(kernelVF.theta) + 1)
    assert(len(kernelVV.theta) == len(kernelFV.theta) + 1)
    assert(len(kernelVV.theta) == len(kernelFF.theta) + 2)
    assert(len(kernelVV.bounds) == len(kernelVF.bounds) + 1)
    assert(len(kernelVV.bounds) == len(kernelFV.bounds) + 1)
    assert(len(kernelVV.bounds) == len(kernelFF.bounds) + 2)

    Rvv, dRvv = kernelVV(G, eval_gradient=True)
    Rvf, dRvf = kernelVF(G, eval_gradient=True)
    Rfv, dRfv = kernelFV(G, eval_gradient=True)
    Rff, dRff = kernelFF(G, eval_gradient=True)

    assert(Rvv == pytest.approx(Rvf))
    assert(Rvv == pytest.approx(Rfv))
    assert(Rvv == pytest.approx(Rff))
    assert(dRvv.shape[2] == dRvf.shape[2] + 1)
    assert(dRvv.shape[2] == dRfv.shape[2] + 1)
    assert(dRvv.shape[2] == dRff.shape[2] + 2)
    assert(dRvv[:, :, kernelVF.active_theta_mask] == pytest.approx(dRvf))
    assert(dRvv[:, :, kernelFV.active_theta_mask] == pytest.approx(dRfv))
    assert(dRvv[:, :, kernelFF.active_theta_mask] == pytest.approx(dRff))


def test_mlgk_kernel_range_check():
    MarginalizedGraphKernel(
        node_kernel=KroneckerDelta(1e-7),
        edge_kernel=TensorProduct(attribute=SquareExponential(1.0))
    )
    MarginalizedGraphKernel(
        node_kernel=TensorProduct(feature=KroneckerDelta(0.5)),
        edge_kernel=TensorProduct(attribute=SquareExponential(1.0))
    )
    with pytest.warns(DeprecationWarning):
        MarginalizedGraphKernel(
            node_kernel=KroneckerDelta(0),
            edge_kernel=TensorProduct(attribute=SquareExponential(1.0))
        )
    with pytest.warns(DeprecationWarning):
        MarginalizedGraphKernel(
            node_kernel=TensorProduct(feature=KroneckerDelta(0.5)) + 1,
            edge_kernel=SquareExponential(1.0)
        )
    with pytest.warns(DeprecationWarning):
        MarginalizedGraphKernel(
            node_kernel=TensorProduct(feature=KroneckerDelta(0.5)),
            edge_kernel=TensorProduct(attribute=SquareExponential(1.0)) + 1
        )
    with pytest.warns(DeprecationWarning):
        MarginalizedGraphKernel(
            node_kernel=KroneckerDelta(0.5) * 2,
            edge_kernel=TensorProduct(attribute=SquareExponential(1.0))
        )
    with pytest.warns(DeprecationWarning):
        MarginalizedGraphKernel(
            node_kernel=TensorProduct(feature=KroneckerDelta(0.5)),
            edge_kernel=TensorProduct(attribute=SquareExponential(1.0)) * 2
        )
