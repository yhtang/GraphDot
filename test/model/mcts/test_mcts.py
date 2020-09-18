#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from rdkit import RDConfig
from rdkit import Chem
from rdkit.Chem import *
from rdkit.Chem import AllChem
from rdkit.Chem import FragmentCatalog
from utils import *
import random
from graphdot import Graph
from graphdot.kernel.marginalized import MarginalizedGraphKernel
from graphdot.model.gaussian_process import GaussianProcessRegressor, LowRankApproximateGPR
from graphdot.graph.reorder import rcm
from graphdot.kernel.fix import Normalization
from graphdot.kernel.basekernel import (
    Normalize,
    Additive,
    TensorProduct,
    Constant,
    Convolution,
    SquareExponential,
    KroneckerDelta
)
from scipy.stats import norm


def test_mcts(molecule, data_path, target, train_size = 100):
    moldata = pd.read_csv(data_path)
    Smiles = moldata['SMILES_str'].tolist()
    KS_gap_old = moldata['e_gap_alpha'].tolist()

    trainX, trainY = [], []
    train_idx = np.random.choice(len(Smiles), train_size, False)
    for i in train_idx:
        try:
            trainX.append(Graph.from_rdkit(Chem.MolFromSmiles(Smiles[i])))
            trainY.append(KS_gap_old[i])
        except:
            pass
    trainX, trainY = np.array([g.permute(rcm(g)) for g in trainX]), np.array(trainY)
    trainX = np.array(trainX)
    trainY = np.array(trainY)

    kernel = MarginalizedGraphKernel(
        Normalize(
            Additive(
                aromatic=Constant(0.5, (0.1, 1.0)) * KroneckerDelta(0.5,(0.1, 0.9)),
                atomic_number=Constant(0.5, (0.1, 1.0)) * KroneckerDelta(0.8,(0.1, 0.9)),
                charge=Constant(0.5, (0.1, 1.0)) * SquareExponential(1.0),
                chiral=Constant(0.5, (0.1, 1.0)) * KroneckerDelta(0.5,(0.1, 0.9)),
                hcount=Constant(0.5, (0.1, 1.0)) * SquareExponential(1.0),
                hybridization=Constant(0.5, (0.1, 1.0)) * KroneckerDelta(0.5,(0.1, 0.9)),
                ring_list=Constant(0.5, (0.01, 1.0)) * Convolution(KroneckerDelta(0.5,(0.1, 0.9)))
            )
        ),
        Normalize(
            Additive(
                aromatic=Constant(0.5, (0.1, 1.0)) * KroneckerDelta(0.5,(0.1, 0.9)),
                conjugated=Constant(0.5, (0.1, 1.0)) * KroneckerDelta(0.5,(0.1, 0.9)),
                order=Constant(0.5, (0.1, 1.0)) * KroneckerDelta(0.8,(0.1, 0.9)),
                ring_stereo=Constant(0.5, (0.1, 1.0)) * KroneckerDelta(0.8,(0.1, 0.9)),
                stereo=Constant(0.5, (0.1, 1.0)) * KroneckerDelta(0.8,(0.1, 0.9))
            )
        ),
        q=0.05
    )
    gp = LowRankApproximateGPR(
        kernel=kernel,
        optimizer=None,
        alpha=1e-5,
        normalize_y=True,
    )
    # gp.kernel.theta = np.array([0, -9.21034037e+00, -2.30258509e+00, -2.30258509e+00, -2.30258509e+00,
    #     0.00000000e+00,  0.00000000e+00, -2.30258509e+00, -1.34987215e-06,
    #    -1.38155106e+01, -2.30258509e+00, -2.30258509e+00, -1.82177446e-04,
    #    -2.30251605e+00,  0.00000000e+00, -1.05360516e-01,  0.00000000e+00,
    #    -1.06280585e-01])
    gp.fit(trainX[:train_size], trainX, trainY, repeat=1, tol=1e-4, verbose=True)

    print("Input Molecule: ", molecule)
    original = gp.predict([Graph.from_rdkit(Chem.MolFromSmiles(molecule))])
    found = False
    while not found:
        mcts = MCTS(gp = gp, molecule=molecule, tree=TreeNode(children=[], parent=None, molecule=molecule, visits=1, allchild=[], depth=0), width = 10, iterations=5, depth=5, target = target, margin = 0.01, sigma_margin = 0.01, exploration_constant = 1)
        found, best_action = mcts.solve()
        molecule = best_action.molecule
        new_predict = gp.predict([Graph.from_rdkit(Chem.MolFromSmiles(molecule))])
        assert (target - original) > (new_predict - original)