#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''An example of similarity comparison between graphs whose node or edge labels
are variable-length sequences rather than scalars using the marginalized graph
kernel.'''
import numpy as np
import networkx as nx
from graphdot import Graph
from graphdot.kernel.marginalized import MarginalizedGraphKernel
from graphdot.kernel.basekernel import (
    TensorProduct,
    Convolution,
    SquareExponential,
    KroneckerDelta
)

# The 'category' attribute on the nodes could have variable lengths.
# So does the 'spectra' attributes on the edges.
g1 = nx.Graph()
g1.add_node(0, category=(1, 2))
g1.add_node(1, category=(2,))
g1.add_edge(0, 1, w=1.0, spectra=[0.5, 0.2])

g2 = nx.Graph()
g2.add_node(0, category=(1, 3))
g2.add_node(1, category=(2, 3, 5))
g2.add_node(2, category=(1,))
g2.add_edge(0, 1, w=2.0, spectra=[0.1, 0.9, 1.5])
g2.add_edge(0, 2, w=0.5, spectra=[0.4])
g2.add_edge(1, 2, w=0.5, spectra=[0.3, 0.6])

# Define node and edge base kernels using the R-convolution framework
# Reference: Haussler, David. Convolution kernels on discrete structures. 1999.
knode = TensorProduct(
    category=Convolution(
        KroneckerDelta(0.5)
    )
)

kedge = TensorProduct(
    spectra=Convolution(
        SquareExponential(0.3)
    )
)

# compose the marginalized graph kernel and compute pairwise similarity
mlgk = MarginalizedGraphKernel(knode, kedge, q=0.05)

R = mlgk([Graph.from_networkx(g, weight='w') for g in [g1, g2]])

# normalize the similarity matrix
d = np.diag(R)**-0.5
K = np.diag(d).dot(R).dot(np.diag(d))

print(K)
