#!/usr/bin/env python
# -*- coding: utf-8 -*-
from rdkit.Chem import MolFromSmiles
from graphdot import Graph
from graphdot.kernel.marginalized import MarginalizedGraphKernel
from graphdot.kernel.marginalized.starting_probability import Uniform
from graphdot.microkernel import (
    Additive,
    Convolution as kConv,
    Constant as kC,
    KroneckerDelta as kDelta,
    SquareExponential as kSE
)
from graphdot.model.gaussian_process import LowRankApproximateGPR


smiles = [
    'CC', 'CCC', 'CCCC', 'CCCCC', 'CCCCCC', 'CCCCCCC', 'CCCCCCCC', 'CCCCCCCCC',
    'CCCCCCCCCC', 'CCCCCCCCCCC', 'CCCCCCCCCCCC'
]
energy = [
    -719.05, -1014.16, -1309.27, -1604.29, -1899.33, -2194.35,
    -2489.38, -2784.41, -3079.44, -3374.47, -3669.50
]

graphs = list(map(lambda smi: Graph.from_rdkit(MolFromSmiles(smi)), smiles))
train_X = graphs[::2]
train_y = energy[::2]
test_X = graphs[1::2]
test_y = energy[1::2]
core = train_X[::2]

kernel = MarginalizedGraphKernel(
    node_kernel=Additive(
        aromatic=kC(0.5, (0.1, 1.0)) * kDelta(0.5, (0.1, 0.9)),
        atomic_number=kC(0.5, (0.1, 1.0)) * kDelta(0.8, (0.1, 0.9)),
        charge=kC(0.5, (0.1, 1.0)) * kSE(1.0),
        chiral=kC(0.5, (0.1, 1.0)) * kDelta(0.5, (0.1, 0.9)),
        hcount=kC(0.5, (0.1, 1.0)) * kSE(1.0),
        hybridization=kC(0.5, (0.1, 1.0)) * kDelta(0.5, (0.1, 0.9)),
        ring_list=kC(0.5, (0.01, 1.0)) * kConv(kDelta(0.5, (0.1, 0.9)))
    ).normalized,
    edge_kernel=Additive(
        aromatic=kC(0.5, (0.1, 1.0)) * kDelta(0.5, (0.1, 0.9)),
        conjugated=kC(0.5, (0.1, 1.0)) * kDelta(0.5, (0.1, 0.9)),
        order=kC(0.5, (0.1, 1.0)) * kDelta(0.8, (0.1, 0.9)),
        ring_stereo=kC(0.5, (0.1, 1.0)) * kDelta(0.8, (0.1, 0.9)),
        stereo=kC(0.5, (0.1, 1.0)) * kDelta(0.8, (0.1, 0.9))
    ).normalized,
    p=Uniform(1.0, (0.1, 40.0)),
    q=0.05
)

gpr = LowRankApproximateGPR(kernel=kernel, alpha=1.0, optimizer=True)
gpr.fit(core, train_X, train_y, verbose=True)
predict_y = gpr.predict(test_X)

print('Prediction:', predict_y)
print('Ground truth:', test_y)
