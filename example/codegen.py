#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''An example of similarity comparison between both node- and edge-labeled,
weighted graphs using the marginalized graph kernel.

This is the scenario that takes full advantage of the marginalized graph
kernel frame.
'''
import numpy as np
import networkx as nx
from graphdot import Graph
from graphdot.kernel.marginalized import MarginalizedGraphKernel
from graphdot.kernel.basekernel import TensorProduct
from graphdot.kernel.basekernel import SquareExponential
from graphdot.kernel.basekernel import KroneckerDelta

# {1.0, 1} --{1.5}[1.0]-- {2.0, 1}
g1 = nx.Graph()
g1.add_node(0, radius=1.0, category=1)
g1.add_node(1, radius=2.0, category=1)
g1.add_edge(0, 1, w=1.0, length=1.5)

# define node and edge kernelets
knode = TensorProduct(radius=SquareExponential(1.0),
                      category=KroneckerDelta(0.5))

kedge = TensorProduct(length=SquareExponential(1.0))

# compose the marginalized graph kernel and compute pairwise similarity
mlgk = MarginalizedGraphKernel(knode, kedge, q=0.05)

R = mlgk([Graph.from_networkx(g1, weight='w')])

# normalize the similarity matrix
d = np.diag(R)**-0.5
K = np.diag(d).dot(R).dot(np.diag(d))

print(K)

print(mlgk.backend.source)
