#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from ase.build import molecule
from graphdot import Graph
from graphdot.kernel.molecular import Tang2019MolecularKernel

# build sample molecules
molecules = [molecule('H2'), molecule('H2O'), molecule('CH4')]

# convert to molecular graphs
graphs = [Graph.from_ase(m) for m in molecules]

# use pre-defined molecular kernel
kernel = Tang2019MolecularKernel()

R = kernel(graphs)

# normalize the similarity matrix
d = np.diag(R)**-0.5
K = np.diag(d).dot(R).dot(np.diag(d))

print(K)
