#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Similarity comparison between molecules whose formula are specified using
the OpenSMILES format.

Since a SMILES string is intrinsically a graph representation of a molecule,
it can be easily used by the marginalized graph kernel.
"""
import numpy as np
import pandas as pd
from graphdot import Graph
from graphdot.kernel.marginalized import MarginalizedGraphKernel
from graphdot.kernel.marginalized.basekernel import TensorProduct
from graphdot.kernel.marginalized.basekernel import SquareExponential
from graphdot.kernel.marginalized.basekernel import KroneckerDelta

# build sample molecules
smiles_list = [
    'CC',  # ethane
    'CCO',  # acetic acid
    'CCN',  # ethylamine
    'C=C',  # ethene
    'CC=C',  # propene
    'CC=CC',  # 2-n-butene
]

# convert to molecular graphs
# nodes(atoms) has 'aromatic', 'charge', 'element', 'hcount' attributes
# edges(bonds) has the 'order' attribute
graphs = [Graph.from_smiles(smi) for smi in smiles_list]

# define node and edge kernelets
knode = TensorProduct(aromatic=KroneckerDelta(0.8, 1.0),
                      charge=SquareExponential(1.0),
                      element=KroneckerDelta(0.5, 1.0),
                      hcount=SquareExponential(1.0))

kedge = TensorProduct(order=KroneckerDelta(0.5, 1.0))

# compose the marginalized graph kernel and compute pairwise similarity
kernel = MarginalizedGraphKernel(knode, kedge, q=0.05)

R = kernel(graphs)

# normalize the similarity matrix and then print
d = np.diag(R)**-0.5
K = np.diag(d).dot(R).dot(np.diag(d))

print(pd.DataFrame(K, columns=smiles_list, index=smiles_list))
