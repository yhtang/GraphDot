#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from graphdot import Graph
from graphdot.microkernel import (
    TensorProduct,
    SquareExponential,
    KroneckerDelta,
)
from graphdot.kernel.marginalized import MarginalizedGraphKernel
from graphdot.kernel.fix import Normalization
from graphdot.metric.maximin import MaxiMin
from ase.build import molecule


np.set_printoptions(linewidth=999, precision=4, suppress=True)

molecules = [
    molecule('CH4'),
    molecule('NH3'),
    molecule('CH3OH'),
    molecule('H2O'),
]

graphs = [Graph.from_ase(m) for m in molecules]

metric = MaxiMin(
    node_kernel=TensorProduct(
        element=KroneckerDelta(0.5)
    ),
    edge_kernel=TensorProduct(
        length=SquareExponential(0.1)
    ),
    q=0.01
)
kernel = Normalization(MarginalizedGraphKernel(
    node_kernel=TensorProduct(
        element=KroneckerDelta(0.5)
    ),
    edge_kernel=TensorProduct(
        length=SquareExponential(0.1)
    ),
    q=0.01
))


def check_hausdorff(X, Y=None):
    # GPU direct computation
    D = metric(X, Y)
    # Manual approach
    K = kernel(X, Y, nodal=True)
    d = np.sqrt(np.maximum(0, 2 - 2 * K))
    starts1 = np.cumsum([0] + [len(g.nodes) for g in X])[:-1]
    starts2 = np.cumsum([0] + [len(g.nodes) for g in Y])[:-1] if Y else starts1
    d1 = np.maximum.reduceat(
        np.minimum.reduceat(
            d, starts2, axis=1
        ),
        starts1, axis=0
    )
    d2 = np.maximum.reduceat(
        np.minimum.reduceat(
            d, starts1, axis=0
        ),
        starts2, axis=1
    )
    d = np.maximum(d1, d2)
    print('-------------------------')
    print(f'GPU direct\n{D}')
    print(f'Manual\n{d}')
    print('-------------------------')


check_hausdorff(graphs[:2], graphs[2:])
check_hausdorff(graphs[:2], graphs[:2])
check_hausdorff(graphs[:2])
check_hausdorff(graphs)
