#!/usr/bin/env python
# -*- coding: utf-8 -*-

from mcts import Rewrite
import os
from rdkit import RDConfig
from rdkit import Chem
from rdkit.Chem import *
from rdkit.Chem import AllChem
from rdkit.Chem import FragmentCatalog
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



class RewriteSmiles(Rewrite): 
    ''' Rewrite rule database. '''

    def __init__(self):
        super(__init__)

    def __call__(self, molecule, random_state):
        ''' Returns a newly rewritten graph. 
        If no seed is provided, the method of rewriting is chosen uniformly between add, substitute, and delete. 
        If a seed is provided, the graph will be modified according to the provided seed value.

        Parameters
        ----------
        graph: string
            The input graph to be rewritten, in string format.
        random_state: float, optional
            The seed value indicating the method of rewriting the graph, from 0 to 1.

        Returns
        -------
        rewrite_graph: string
            The newly rewritten graph in string format.
        
        '''
        np.random.seed(random_state)
        possible_actions = Rewrites.getValidActions(graph)[np.random.randint(2)]
        action = np.array(possible_actions)[np.random.choice(len(possible_actions), min(len(possible_actions),self.width), False)]
        return action
        # addlst, sublst, remlst = Rewrites.getValidActions(graph)
        # possible_actions = addlst + sublst + remlst
        # possible_actions = np.array(possible_actions)[np.random.choice(len(possible_actions), min(len(possible_actions),self.width), False)]
    
    def _process(smiles):
        '''' Gets rid of all the temporary characters in a SMILES string. '''
        smiles = smiles.replace("*", "")
        smiles = smiles.replace("#", "")
        smiles = smiles.replace(".", "")
        return smiles
    
    def _add(smiles, index, bondtype, new):
        ''' Adds a new atom to the molecule. 
        Parameters
        ----------
            smiles: string, the SMILES string of the molecule
            index: int, where to add the new atom
            bondtype: string, either "Single" or "Double"
            new: int, atomic number of atom to be added in
        Returns
        -------
            molecule: graph, current molecule being produced, None if not possible
        '''
        if bondtype == 'Single':
            bond = Chem.BondType.SINGLE
        elif bondtype == 'Double':
            bond = Chem.BondType.DOUBLE
        try:
            m = Chem.MolFromSmiles(smiles)
            length = m.GetNumAtoms()
            mw = Chem.RWMol(m)
            mw.AddAtom(Chem.Atom(new))
            mw.AddBond(index, length, bond)
            return Chem.MolToSmiles(mw)
        except:
            return None
        
    def _substitute(smiles, original, replacement):
        ''' Substitutes a subgraph with another subgraph.

        Parameters
        ----------
        smiles: string
            The full graph being rewritten.
        original: string
            The subgraph being taken out.
        replacement: string
            The new subgraph being substituted in.
            
        Returns
        -------
        graph_lst: list
            Returns the list of graphs that can be produced with this operation, and None if not possible.
            Note that more than one graph can be produced because the given subgraph may occur in multiple
            parts of the original graph. 
        '''
        try:
            molecule = Chem.MolFromSmiles(smiles)
            o = Chem.MolFromSmarts(original)
            r = Chem.MolFromSmarts(replacement)
            new = AllChem.ReplaceSubstructs(molecule, o, r)
            return [Rewrites._process(Chem.MolToSmiles(m)) for m in new]
        except: 
            return None

    def _delete(smiles, substruct):
        ''' Deletes the given subgraph from the original graph.

        Parameters
        ----------
        smiles: string
            The full graph being rewritten.
        substruct: string
            The subgraph being taken out.

        Returns
        ------
        graph_lst: list
            Returns the list of graphs that can be produced with this operation, and None if not possible.
            More than one graph can be produced because the given subgraph may occur in multiple
            parts of the original graph. 
        '''
        molecule = Chem.MolFromSmiles(smiles)
        sub = Chem.MolFromSmarts(substruct)
        try: 
            new = AllChem.DeleteSubstructs(molecule,sub)
            return Rewrites._process(Chem.MolToSmiles(new))
        except: 
            return None
    
    def getFuncGroups(smiles):
        ''' Returns a tuple of (name, SMILES) of all the functional groups present in a molecule. 
        Parameters:
            smiles: string, SMILES representation of molecule to be tested
        Returns:
            result: (string, string), tuple of form (name, SMILES)
        '''
        try:
            fName=os.path.join('Data/FunctionalGroups/FunctionalGroups.txt')
            fparams = FragmentCatalog.FragCatParams(1,6,fName)
            fcat=FragmentCatalog.FragCatalog(fparams)
            fcgen=FragmentCatalog.FragCatGenerator()
            m = Chem.MolFromSmiles(smiles)
            fcgen.AddFragsFromMol(m,fcat)
            fs = list(fcat.GetEntryFuncGroupIds(fcat.GetNumEntries() - 1))
            result = []
            for idx in fs:
                mol = fparams.GetFuncGroup(idx)
                s = Chem.MolToSmiles(mol)
                name = mol.GetProp('_Name')
                result.append((name, s))
            return result
        except: 
            return []
    
    def allFuncGroups():
        ''' Returns lists of all functional groups in RDKit.
        Returns:
            names: list(string), names of all functional groups
            smiles: list(string), names of all SMILES representations of functional groups
        '''
        fName=os.path.join('Data/FunctionalGroups/FunctionalGroups.txt')
        fparams = FragmentCatalog.FragCatParams(1,6,fName)
        txt = fparams.Serialize().replace("\t","\n").split("\n")[2:-1]
        names = [txt[i] for i in range(0,len(txt),2)]
        smiles = [txt[i] for i in range(1,len(txt),2)]
        return names, smiles
    
    def getValidActions(mol):
        ''' Returns two lists of all possible add and substitute actions that can be made.
        Parameters: 
            mol: string, SMILES representation of molecule to be tested
        Returns:
            addlst: list, with each item being of form (result, location, bondType, atom)
            sublst: list, with each item being of form (result, original, replacement)
            remlst: list, with each item being of form (result, original)
        '''
        elementlst = [6, 7, 8, 9, 15, 16, 17, 35, 53] # C, N, O, P, S and the Halogens
        addlst = []
        sublst = []
        remlst = []
        repeats = {}
        funcgroups = Rewrites.getFuncGroups(mol)
        if funcgroups is not None:
            for (name, smiles) in funcgroups:
                namelst, smileslst = Rewrites.allFuncGroups()
                for repl in smileslst:
                    sub = Rewrites._substitute(mol, smiles, repl)
                    for elem in sub:
                        if elem not in repeats:
                            sublst.append((elem, smiles, repl))
                            repeats[elem] = 1
                rem = Rewrites._delete(mol, smiles)
                if rem is not None and rem not in repeats: 
                    remlst.append((rem, smiles))
                    repeats[rem] = 1
            for element in elementlst:
                for i in range(len(mol)):
                    for bondType in ['Single', 'Double']:
                        add = Rewrites._add(mol, i, bondType, element)
                        if add is not None and add not in repeats: 
                            addlst.append((add, i, bondType, element))
                            repeats[add] = 1
            return addlst, sublst, remlst
        else:
            return None, None

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
        mcts = MCTS(predictor = gp, seed_graph=molecule, tree=TreeNode(children=[], parent=None, graph=molecule, visits=1, allchild=[], depth=0), width = 10, iterations=5, depth=5, target = target, margin = 0.01, sigma_margin = 0.01, exploration_constant = 1)
        found, best_action = mcts.solve()
        molecule = best_action.graph
        new_predict = gp.predict([Graph.from_rdkit(Chem.MolFromSmiles(molecule))])
        assert (target - original) > (new_predict - original)