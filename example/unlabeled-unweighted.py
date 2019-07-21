#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''An example of similarity comparison between unlabeled and unweighted graphs
using the marginalized graph kernel.

Note: it is known that all unlabeled and unweighted graphs are identical under
the similarity metric defined by the marginalized graph kernel. This example
merely illustrates the usage of the package.
'''
import numpy as np
import networkx as nx
from graphdot import Graph
from graphdot.kernel.marginalized import MarginalizedGraphKernel
from graphdot.kernel.marginalized.basekernel import Constant

# simple graph: 0 -- 1
graph1 = nx.Graph()
graph1.add_node(0)
graph1.add_node(1)
graph1.add_edge(0, 1)

# simple graph: 0 -- 1 -- 2
graph2 = nx.Graph()
graph2.add_node(0)
graph2.add_node(1)
graph2.add_node(2)
graph2.add_edge(0, 1)
graph2.add_edge(1, 2)

# simple graph: 0 --- 1
#                \  /
#                 2
graph3 = nx.Graph()
graph3.add_node(0)
graph3.add_node(1)
graph3.add_node(2)
graph3.add_edge(0, 1)
graph3.add_edge(0, 2)
graph3.add_edge(1, 2)

# define trivial node and edge kernelets
knode = Constant(1.0)
kedge = Constant(1.0)

# compose the marginalized graph kernel and compute pairwise similarity
mlgk = MarginalizedGraphKernel(knode, kedge, q=0.05)

R = mlgk([Graph.from_networkx(g) for g in [graph1, graph2, graph3]])

# normalize the similarity matrix
d = np.diag(R)**-0.5
K = np.diag(d).dot(R).dot(np.diag(d))

# all entries should be approximately 1 plus round-off error
print(K)
