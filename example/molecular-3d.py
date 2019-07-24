#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Similarity comparison between molecular configurations in 3D.

The molecules are first converted to molecular graphs using an 'adjacency rule'
as described in Tang & de Jong https://doi.org/10.1063/1.5078640, then computed
using the marginalized graph kernel.
"""
import numpy as np
import pandas as pd
from ase.build import molecule, bulk
from graphdot import Graph
from graphdot.kernel.molecular import Tang2019MolecularKernel

# build sample molecules
small_title = ['H2O', 'HCl', 'NaCl']
bulk_title = ['NaCl-bulk', 'NaCl-bulk2']
bulk = [
    bulk('NaCl', 'rocksalt', a=5.64),
    bulk('NaCl', 'rocksalt', a=5.66),
]
molecules = [molecule(name) for name in small_title] + bulk

# convert to molecular graphs
graphs = [Graph.from_ase(m) for m in molecules]

# use pre-defined molecular kernel
kernel = Tang2019MolecularKernel(edge_length_scale=0.1)

R = kernel(graphs)

# normalize the similarity matrix
d = np.diag(R)**-0.5
K = np.diag(d).dot(R).dot(np.diag(d))

# note the difference between the NaCl variants
title = small_title + bulk_title
print(pd.DataFrame(K, columns=title, index=title))
