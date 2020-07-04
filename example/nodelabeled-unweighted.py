#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''An example of similarity comparison between node-labeled but unweighted
graphs using the marginalized graph kernel.'''
import numpy as np
import networkx as nx
from graphdot import Graph
from graphdot.kernel.marginalized import MarginalizedGraphKernel
from graphdot.microkernel import (
    TensorProduct,
    SquareExponential,
    KroneckerDelta,
    Constant
)

# {1.0, 1} -- {2.0, 1}
g1 = nx.Graph()
g1.add_node(0, radius=1.0, category=1)
g1.add_node(1, radius=2.0, category=1)
g1.add_edge(0, 1)

# {1.0, 1} -- {2.0, 1} -- {1.0, 2}
g2 = nx.Graph()
g2.add_node(0, radius=1.0, category=1)
g2.add_node(1, radius=2.0, category=1)
g2.add_node(2, radius=1.0, category=2)
g2.add_edge(0, 1)
g2.add_edge(1, 2)

# {1.0, 1} -- {2.0, 1}
#     \         /
#      {1.0, 2}
g3 = nx.Graph()
g3.add_node(0, radius=1.0, category=1)
g3.add_node(1, radius=2.0, category=1)
g3.add_node(2, radius=1.0, category=2)
g3.add_edge(0, 1)
g3.add_edge(0, 2)
g3.add_edge(1, 2)

# define node and edge kernelets
knode = TensorProduct(radius=SquareExponential(0.5),
                      category=KroneckerDelta(0.5))

kedge = Constant(1.0)

# compose the marginalized graph kernel and compute pairwise similarity
mlgk = MarginalizedGraphKernel(knode, kedge, q=0.05)

R = mlgk([Graph.from_networkx(g) for g in [g1, g2, g3]])

# normalize the similarity matrix
d = np.diag(R)**-0.5
K = np.diag(d).dot(R).dot(np.diag(d))

print(K)
