#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''An example of similarity comparison between graphs whose node or edge labels
are variable-length sequences rather than scalars using the marginalized graph
kernel.'''
import networkx as nx
from graphdot import Graph
from graphdot.kernel.marginalized import MarginalizedGraphKernel
from graphdot.kernel.fix import Normalization
from graphdot.microkernel import (
    TensorProduct,
    DotProduct,
    Constant
)

# The 'category' attribute on the nodes could have variable lengths.
# So does the 'spectra' attributes on the edges.
g1 = nx.Graph()
g1.add_node(0, soap=[0.5, 1.5, 2.5, 0.5])
g1.add_node(1, soap=[0.5, 1.5, 2.5, 0.5])
g1.add_edge(0, 1, w=1.0)

g2 = nx.Graph()
g2.add_node(0, soap=[0.5, 1.5, 2.5, 3.5])
g2.add_node(1, soap=[1.5, 1.5, 0.5, 3.5])
g2.add_node(2, soap=[0.5, 2.5, 2.5, 0.5])
g2.add_edge(0, 1, w=2.0)
g2.add_edge(0, 2, w=0.5)
g2.add_edge(1, 2, w=0.5)

# compose the marginalized graph kernel and compute pairwise similarity
mlgk = Normalization(
    MarginalizedGraphKernel(
        node_kernel=TensorProduct(
            soap=DotProduct().normalized
        ),
        edge_kernel=Constant(1),
        q=0.05
    )
)

G = [Graph.from_networkx(g, weight='w') for g in [g1, g2]]
print(f'Whole-graph similarity\n{mlgk(G)}')
print(f'Nodal similarity\n{mlgk(G, nodal=True)}')
