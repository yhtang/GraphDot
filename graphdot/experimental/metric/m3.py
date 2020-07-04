#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
from graphdot.graph import Graph
from graphdot.graph.adjacency.atomic import AtomicAdjacency
from graphdot.microkernel import (
    TensorProduct,
    KroneckerDelta,
    SquareExponential
)


class M3:
    """The Marginalized MiniMax (M3) metric between molecules"""

    def __init__(self, use_charge=False, adjacency='default', q=0.01,
                 element_delta=0.2, bond_eps=0.02, charge_eps=0.2):

        self.use_charge = use_charge
        if adjacency == 'default':
            self.adjacency = AtomicAdjacency(shape='tent2', zoom=0.75)
        else:
            self.adjacency = adjacency
        self.q = q
        if use_charge:
            self.node_kernel = TensorProduct(
                element=KroneckerDelta(element_delta),
                charge=SquareExponential(charge_eps),
            )
        else:
            self.node_kernel = TensorProduct(
                element=KroneckerDelta(element_delta)
            )
        self.edge_kernel = TensorProduct(length=SquareExponential(bond_eps))

    def __call__(self, atoms1, atoms2):
        args = dict(use_charge=self.use_charge, adjacency=self.adjacency)
        g1 = Graph.from_ase(atoms1, **args)
        g2 = Graph.from_ase(atoms2, **args)

        R1 = self._mlgk(g1, g1).diagonal()**-0.5
        R2 = self._mlgk(g2, g2).diagonal()**-0.5
        R12 = self._mlgk(g1, g2)

        K = R1[:, None] * R12 * R2[None, :]
        D = np.sqrt(np.maximum(2 - 2 * K, 0))

        return max(D.min(axis=1).max(), D.min(axis=0).max())

    def _mlgk(self, g1, g2):
        n1 = len(g1.nodes)
        n2 = len(g2.nodes)

        A1 = scipy.sparse.csc_matrix(
            (g1.edges['!w'], (g1.edges['!i'], g1.edges['!j'])),
            (n1, n1)
        )
        A2 = scipy.sparse.csc_matrix(
            (g2.edges['!w'], (g2.edges['!i'], g2.edges['!j'])),
            (n2, n2)
        )
        A1 += A1.T
        A2 += A2.T

        d1 = np.asarray(A1.sum(axis=0)).ravel()
        d2 = np.asarray(A2.sum(axis=0)).ravel()

        Ax = scipy.sparse.kron(A1, A2)

        Vx = np.array(
            [self.node_kernel(atom1, atom2)
             for atom1 in g1.nodes.itertuples()
             for atom2 in g2.nodes.itertuples()])

        E, i, j = [], [], []
        for i1, j1, e1 in zip(g1.edges['!i'],
                              g1.edges['!j'],
                              g1.edges.itertuples()):
            for i2, j2, e2 in zip(g2.edges['!i'],
                                  g2.edges['!j'],
                                  g2.edges.itertuples()):
                e = self.edge_kernel(e1, e2)
                E += [e, e, e, e]
                i += [i1 * n2 + i2,
                      j1 * n2 + i2,
                      j1 * n2 + j2,
                      i1 * n2 + j2]
                j += [j1 * n2 + j2,
                      i1 * n2 + j2,
                      i1 * n2 + i2,
                      j1 * n2 + i2]
        Ex = scipy.sparse.csc_matrix((E, (i, j)), (n1 * n2, n1 * n2))

        qx = np.kron(np.ones(n1), np.ones(n2))
        Dx = np.kron(d1, d2) / (1 - self.q)**2
        rhs = Dx * qx
        Y = scipy.sparse.diags([Dx / Vx], [0]) - Ax.multiply(Ex)
        R, _ = scipy.sparse.linalg.cg(
            Y, rhs,
            M=scipy.sparse.diags([Vx / Dx], [0]),
            atol=1e-7
        )

        return R.reshape(n1, n2)
