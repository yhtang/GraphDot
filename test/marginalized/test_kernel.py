#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pytest
import networkx as nx
from graphdot import Graph
from graphdot.marginalized import MarginalizedGraphKernel
from graphdot.marginalized.basekernel import Constant
from graphdot.marginalized.basekernel import KroneckerDelta
from graphdot.marginalized.basekernel import SquareExponential
from graphdot.marginalized.basekernel import TensorProduct


# g1 = nx.Graph(title='A')
# g1.add_node('a', type=0)
# g1.add_node('b', type=0)
# g1.add_edge('a', 'b', weight=1.0)
#
# node_kernel = TensorProduct(type=KroneckerDelta(1.0, 1.0))
# edge_kernel = Constant(1.0)
#
# q = 0.5
# mlgk = MarginalizedGraphKernel(node_kernel, edge_kernel, q=q)
#
# R = mlgk.compute([Graph.from_networkx(g, weight='weight') for g in [g1]])
#
# D = np.kron(np.eye(2) / (1-q), np.eye(2) / (1-q))
# A = np.kron(1-np.eye(2), 1-np.eye(2))
# r = np.kron(np.ones(2) * q, np.ones(2) * q)
# rhs = D.dot(r)
# linsys = D - A
# solution = np.linalg.solve(linsys, rhs)
# print('linsys\n', linsys, sep='')
# print('rhs\n', rhs, sep='')
# print('solution\n', solution, sep='')
#
# print(R)

# def test_mlgk_vanilla():
#
#     g1 = nx.Graph(title='A')
#     g1.add_node('a', type=0)
#     g1.add_node('b', type=0)
#     g1.add_edge('a', 'b', weight=1.0)
#
#     # g2 = nx.Graph(title='B')
#     # g2.add_node('a', type=0)
#     # g2.add_node('b', type=0)
#     # g2.add_edge('a', 'b', weight=1.0)
#
#     node_kernel = TensorProduct(type=KroneckerDelta(1.0, 1.0))
#     edge_kernel = Constant(1.0)
#
#     mlgk = MarginalizedGraphKernel(node_kernel, edge_kernel)
#
#     R = mlgk.compute([Graph.from_networkx(g, weight='weight') for g in [g1]])
#
#     print(R)

# def test_mlgk():
#
#     class Hybrid:
#         NONE = np.int32(0)
#         SP = np.int32(1)
#         SP2 = np.int32(2)
#         SP3 = np.int32(3)
#
#     g1 = nx.Graph(title='H2O')
#     g1.add_node('O1', hybridization=Hybrid.SP2, charge=np.int32(1))
#     g1.add_node('H1', hybridization=Hybrid.SP3, charge=np.int32(-1))
#     g1.add_node('H2', hybridization=Hybrid.SP, charge=np.int32(2))
#     g1.add_edge('O1', 'H1', order=np.int32(1), length=np.float32(0.5))
#     g1.add_edge('O1', 'H2', order=np.int32(2), length=np.float32(1.0))
#
#     g2 = nx.Graph(title='H2')
#     g2.add_node('H1', hybridization=Hybrid.SP, charge=np.int32(1))
#     g2.add_node('H2', hybridization=Hybrid.SP, charge=np.int32(1))
#     g2.add_edge('H1', 'H2', order=np.int32(2), length=np.float32(1.0))
#
#     node_kernel = TensorProduct(hybridization=KroneckerDelta(0.3, 1.0),
#                                 charge=SquareExponential(1.0))
#
#     edge_kernel = TensorProduct(order=KroneckerDelta(0.3, 1.0),
#                                 length=SquareExponential(0.05))
#
#     mlgk = MarginalizedGraphKernel(node_kernel, edge_kernel)
#
#     R = mlgk.compute([Graph.from_networkx(g) for g in [g1, g2]])
#
#     assert(R.shape == (2, 2))
#     assert(np.count_nonzero(R - R.T) == 0)
#
#
# def test_mlgk_weighted():
#
#     class Hybrid:
#         NONE = np.int32(0)
#         SP = np.int32(1)
#         SP2 = np.int32(2)
#         SP3 = np.int32(3)
#
#     g1 = nx.Graph(title='H2O')
#     g1.add_node('O1', hybridization=Hybrid.SP2, charge=np.int32(1))
#     g1.add_node('H1', hybridization=Hybrid.SP3, charge=np.int32(-1))
#     g1.add_node('H2', hybridization=Hybrid.SP, charge=np.int32(2))
#     g1.add_edge('O1', 'H1', order=np.int32(1), length=np.float32(0.5), w=1.0)
#     g1.add_edge('O1', 'H2', order=np.int32(2), length=np.float32(1.0), w=2.0)
#
#     g2 = nx.Graph(title='H2')
#     g2.add_node('H1', hybridization=Hybrid.SP, charge=np.int32(1))
#     g2.add_node('H2', hybridization=Hybrid.SP, charge=np.int32(1))
#     g2.add_edge('H1', 'H2', order=np.int32(2), length=np.float32(1.0), w=3.0)
#
#     node_kernel = TensorProduct(hybridization=KroneckerDelta(0.3, 1.0),
#                                 charge=SquareExponential(1.0))
#
#     edge_kernel = TensorProduct(order=KroneckerDelta(0.3, 1.0),
#                                 length=SquareExponential(0.05))
#
#     mlgk = MarginalizedGraphKernel(node_kernel, edge_kernel)
#
#     R = mlgk.compute([Graph.from_networkx(g, weight='w') for g in [g1, g2]])
#
#     assert(R.shape == (2, 2))
#     assert(np.count_nonzero(R - R.T) == 0)
