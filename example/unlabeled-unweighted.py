#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''An example of similarity comparison between unlabeled and unweighted graphs
using the marginalized graph kernel.

Note: it is known that all unlabeled and unweighted graphs are identical under
the similarity metric defined by the marginalized graph kernel. This example
merely illustrates the usage of the package.'''
import numpy as np
import networkx as nx
from graphdot import Graph
from graphdot.kernel.marginalized import MarginalizedGraphKernel
from graphdot.kernel.marginalized.basekernel import Constant

# 0 -- 1
g1 = nx.Graph()
g1.add_node(0)
g1.add_node(1)
g1.add_edge(0, 1)

# 0 -- 1 -- 2
g2 = nx.Graph()
g2.add_node(0)
g2.add_node(1)
g2.add_node(2)
g2.add_edge(0, 1)
g2.add_edge(1, 2)

# 0 --- 1
#  \  /
#   2
g3 = nx.Graph()
g3.add_node(0)
g3.add_node(1)
g3.add_node(2)
g3.add_edge(0, 1)
g3.add_edge(0, 2)
g3.add_edge(1, 2)

# define trivial node and edge kernelets
knode = Constant(1.0)
kedge = Constant(1.0)

# compose the marginalized graph kernel and compute pairwise similarity
mlgk = MarginalizedGraphKernel(knode, kedge, q=0.05)

R = mlgk([Graph.from_networkx(g) for g in [g1, g2, g3]])

# normalize the similarity matrix
d = np.diag(R)**-0.5
K = np.diag(d).dot(R).dot(np.diag(d))

# all entries should be approximately 1 plus round-off error
print(K)
