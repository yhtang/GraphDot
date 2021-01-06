#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pytest
from ase.build import molecule
from graphdot import Graph
from graphdot.metric.maximin import MaxiMin
from graphdot.microkernel import (
    KroneckerDelta,
    SquareExponential,
    TensorProduct,
)


G = [Graph.from_ase(molecule(f)) for f in ['CH3SCH3', 'CH3OCH3']]
H = [Graph.from_ase(molecule(f)) for f in ['CH4', 'NH3', 'H2O']]


def test_maximin_basic():
    metric = MaxiMin(
        node_kernel=TensorProduct(
            element=KroneckerDelta(0.5)
        ),
        edge_kernel=TensorProduct(
            length=SquareExponential(0.1)
        ),
        q=0.01
    )
    distance = metric(G)
    assert distance.shape == (len(G), len(G))
    assert np.allclose(distance.diagonal(), 0, atol=1e-3)
    assert np.all(distance >= 0)
    assert np.allclose(distance, distance.T, rtol=1e-14, atol=1e-14)

    distance = metric(G, G)
    assert distance.shape == (len(G), len(G))
    assert np.allclose(distance.diagonal(), 0, atol=1e-3)
    assert np.all(distance >= 0)
    assert np.allclose(distance, distance.T, rtol=1e-4, atol=1e-4)

    distance = metric(G, H)
    assert distance.shape == (len(G), len(H))
    assert np.all(distance >= 0)


# def test_maximin_hotspot():
#     g = [Graph.from_ase(molecule(f)) for f in ['CH3SCH3', 'CH3OCH3']]
#     print(g[0].nodes.to_pandas())
#     print(g[1].nodes.to_pandas())

#     metric = MaxiMin(
#         node_kernel=TensorProduct(
#             element=KroneckerDelta(0.5)
#         ),
#         edge_kernel=TensorProduct(
#             length=SquareExponential(0.1)
#         ),
#         q=0.01
#     )

#     from graphdot.kernel.fix import Normalization
#     from graphdot.kernel.marginalized import MarginalizedGraphKernel
#     kernel = Normalization(MarginalizedGraphKernel(
#         node_kernel=TensorProduct(
#             element=KroneckerDelta(0.5)
#         ),
#         edge_kernel=TensorProduct(
#             length=SquareExponential(0.1)
#         ),
#         q=0.01
#     ))
#     K = kernel([g[0]], [g[1]], nodal=True)
#     np.set_printoptions(precision=4, linewidth=999, suppress=True)
#     print(f'K\n{K}')
#     D = np.sqrt(np.maximum(0, 2 - 2 * K))
#     print(f'D\n{D}')
#     print(D.min(axis=1))
#     print(D.min(axis=0))

#     index_S = np.flatnonzero(g[0].nodes.element == 16)
#     index_O = np.flatnonzero(g[1].nodes.element == 8)
#     distance, (I, J) = metric(g, return_hotspot=True)
#     print(f'distance\n{distance}')

#     print(I)
#     print(J)
#     assert I.shape == (2, 2)
#     assert J.shape == (2, 2)
#     assert I[0, 1] == index_S


@pytest.mark.parametrize('X', [(G,), (G, H), ([G[0], G[1]], [H[0], G[1]])])
def test_maximin_gradient(X):
    metric = MaxiMin(
        node_kernel=TensorProduct(
            element=KroneckerDelta(0.5)
        ),
        edge_kernel=TensorProduct(
            length=SquareExponential(0.1)
        ),
        q=0.01
    )

    _, gradient = metric(*X, eval_gradient=True)
    if len(X) == 1:
        n1 = n2 = len(X[0])
    elif len(X) == 2:
        n1 = len(X[0])
        n2 = len(X[1])
    assert gradient.shape == (n1, n2, len(metric.theta))
    eps = 1e-2
    for k, _ in enumerate(metric.theta):
        pos = np.exp(metric.theta)
        neg = np.exp(metric.theta)
        pos[k] += eps
        neg[k] -= eps
        distance_pos = metric.clone_with_theta(np.log(pos))(*X)
        distance_neg = metric.clone_with_theta(np.log(neg))(*X)
        grad_diff = (distance_pos - distance_neg) / (2 * eps)
        if len(X) == 1:
            for i in range(n1):
                for j in range(n1):
                    if i == j:
                        assert gradient[i, j, k] == pytest.approx(
                            0, abs=0.05, rel=0.05
                        )
                    else:
                        assert gradient[i, j, k] == gradient[j, i, k]
                        assert gradient[i, j, k] == pytest.approx(
                            grad_diff[i, j], abs=0.05, rel=0.05
                        )
        elif len(X) == 2:
            assert np.allclose(
                gradient[:, :, k], grad_diff, atol=0.05, rtol=0.05
            )
