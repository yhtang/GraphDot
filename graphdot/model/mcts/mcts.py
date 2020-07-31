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

train_size = 100
moldata = read_cepdb()
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
gp.fit(trainX[:100], trainX, trainY, repeat=1, tol=1e-4, verbose=True)

class Rewrites: 
    ''' Rewrite rule database. '''
    
    def _process(smiles):
        '''' Gets rid of all the temporary characters in a SMILES string. '''
        smiles = smiles.replace("*", "")
        smiles = smiles.replace("#", "")
        smiles = smiles.replace(".", "")
        return smiles
    
    def add(smiles, index, bondtype, new):
        ''' Adds a new atom to the molecule. 
        Parameters: 
            smiles: string, the SMILES string of the molecule
            index: int, where to add the new atom
            bondtype: string, either "Single" or "Double"
            new: int, atomic number of atom to be added in
        Returns:
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
        
    def substitute(smiles, original, replacement):
        ''' Switches given substruct with another substruct. 
        Parameters: 
            smiles: string, the SMILES string of the molecule
            original: string, original SMARTS pattern to be switched out
            replacement: string, replacement SMARTS pattern to be switched in
        Returns:
            molecule_lst: list of graphs, current molecule being produced, None if not possible
        '''
        try:
            molecule = Chem.MolFromSmiles(smiles)
            o = Chem.MolFromSmarts(original)
            r = Chem.MolFromSmarts(replacement)
            new = AllChem.ReplaceSubstructs(molecule, o, r)
            return [Rewrites._process(Chem.MolToSmiles(m)) for m in new]
        except: 
            return None

    def delete(smiles, substruct):
        ''' Deleted given substruct. 
        Parameters: 
            smarts: string, the SMILES string of the molecule
            substruct: string, original SMARTS pattern to be deleted
        Returns:
            molecule_lst: list of graphs, current molecule being produced, None if not possible
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
                    sub = Rewrites.substitute(mol, smiles, repl)
                    for elem in sub:
                        if elem not in repeats:
                            sublst.append((elem, smiles, repl))
                            repeats[elem] = 1
                rem = Rewrites.delete(mol, smiles)
                if rem is not None and rem not in repeats: 
                    remlst.append((rem, smiles))
                    repeats[rem] = 1
            for element in elementlst:
                for i in range(len(mol)):
                    for bondType in ['Single', 'Double']:
                        add = Rewrites.add(mol, i, bondType, element)
                        if add is not None and add not in repeats: 
                            addlst.append((add, i, bondType, element))
                            repeats[add] = 1
            return addlst, sublst, remlst
        else:
            return None, None

class TreeNode:
    ''' A MCTS tree node representing a new molecule.
    Parameters: 
        children: list, list of TreeNodes of its children
        parent: TreeNode, parent TreeNode or None (for the root node)
        molecule: string, current Smolecule being produced
        visits: integer, number of times this node has been visited
        allchild: list of Graphs, list of all molecules in its subtree
        depth: integer, depth of molecule in the tree
    '''
    def __init__(self, children, parent, molecule, visits, allchild, depth):
        self.children = children
        self.parent = parent
        self.molecule = molecule
        self.visits = visits
        self.allchild = allchild
        self.mean = 0
        self.std = 0
        self.post_mean = 0
        self.post_var = 0
        self.mlt = 0
        self.depth = depth
        self.selected = False

    def __repr__(self):
        if self.children:
            branch_str = ', ' + repr(self.children)
        else:
            branch_str = ''
        return 'Node({0}{1})'.format(self.molecule, branch_str)

    def __str__(self):
        def print_tree(t, indent=0):
            space = 50 - (t.depth*4 + len(t.molecule))
            if t.selected: select = 1
            else: select = 0
            if t.children:
                tree_str = '    ' * indent + '*' * select + t.molecule + " " * space + "(Post_mean: " + str(t.post_mean)[:5] + " Post_var: " + str(t.post_var)[:5] + " MLT: " + str(t.mlt)[:5] + ")\n"
                for child in t.children:
                    tree_str += print_tree(child, indent + 1)
            else:
                tree_str = '    ' * indent + '*' * select + t.molecule + " " * space + "(Mean: " + str(t.mean)[:5] + " Std: " + str(t.std)[:5] + " MLT: " + str(t.mlt)[:5] + ")\n"
            return tree_str
        return print_tree(self).rstrip()

class MCTS:
    ''' Monte Carlo Tree Search algorithm. 
    Parameters: 
        gp: GaussianProcessRegressor instance
        molecule: graph, current molecule being produced
        tree: TreeNode representing the current state of the MCTS tree
        width: integer, number of maximum nodes allowed in one level
        iterations: integer, number of MCTS iterations to run in each turn
        depth: integer, maximum depth allowed in node selection
        target: float, target H/L gap value
        margin: float, margin of error for target value that can be accepted
        sigma_margin: float, margin of error for sigma value that can be accepted
        exploration_constant: float, hyperparameter to be tuned
    '''
    
    def __init__(self, gp, molecule, tree, width, iterations, depth, target, margin, sigma_margin, exploration_constant):
        self.molecule = molecule
        self.tree = tree
        self.iterations = iterations
        self.depth = depth
        self.width = width
        self.target = target
        self.gp = gp
        self.margin = margin
        self.sigma_margin = sigma_margin
        self.exploration_constant = exploration_constant
    
    def check_found(self, child, pred, sigma):
        ''' Checks to see if a molecule has been found.
        Parameters:
            child: TreeNode, node to check if the molecule has been found
            pred: float, prediction of the molecule's H/L gap
            sigma: float, the molecule's standard deviation
        Returns:
            found: boolean, whether or not the molecule has properties within the acceptable margin of error
        '''
        if sigma < self.sigma_margin and abs(pred - self.target) < self.margin:
            child.selected = True
            return True
        return False
    
    def selection(self): 
        ''' Select leaf node with maximum UCT value. 
        Returns: 
            max_node: TreeNode, node with maximum UCT value
            depth: integer, depth of the max_node found from the root node
        '''
        max_node = self.tree
        max_mlt = self.tree.mlt
        depth = 0
        while len(max_node.children) != 0: 
            prev_max = max_node
            depth += 1
            for child in max_node.children: 
                if child.mlt > max_mlt:
                    max_mlt = child.mlt
                    max_node = child
            if max_node == prev_max:
                children = np.array(max_node.children)
                max_node = children[np.random.randint(0, len(children), 1)[0]]
        return max_node, depth
    
    def expansion(self, node):
        ''' Create all possible outcomes from leaf node. 
        Parameters:
            node: TreeNode, maximum UCT node found by selection()
        Returns:
            created: boolean, whether or not new children were added
        '''
        molecule = node.molecule
        addlst, sublst, remlst = Rewrites.getValidActions(molecule)
        possible_actions = addlst + sublst + remlst
        possible_actions = np.array(possible_actions)[np.random.choice(len(possible_actions), min(len(possible_actions),self.width), False)]
        for action in possible_actions: 
            try:
                new_mol = action[0]
                Graph.from_rdkit(Chem.MolFromSmiles(new_mol))
                child = TreeNode(children=[], parent=node, molecule=new_mol, visits=0, allchild=[], depth=node.depth+1)
                node.children.append(child)
            except:
                pass
        return node.children
                
    def simulation(self, node):
        ''' Simulate game from child node's state until it reaches the resulting state of the game. 
        Parameters:    
            node: TreeNode, child node found by expansion()
        Returns: 
            result: boolean, representing whether or not a molecule was found
            child: TreeNode, node of the molecule found
            graphs: list, list of graphs of all the node's children
        '''
        molecule = node.molecule
        graphs = [Graph.from_rdkit(Chem.MolFromSmiles(child.molecule)) for child in node.children]
        mean, cov = self.gp.predict(graphs, return_cov=True)
        for i in range(len(node.children)):
            child = node.children[i]
            child.mean = mean[i]
            child.std = np.sqrt(cov[i][i])
            exploitation = norm(child.mean, child.std).pdf(self.target)
            exploration = np.sqrt(self.tree.visits)
            child.mlt = exploitation + self.exploration_constant * exploration
            child.visits += 1
            if self.check_found(child, child.mean, child.std):
                return True, child, graphs
        return False, None, graphs
    
    def backpropagation(self, node, allchild_graphs):
        ''' Backpropagate against the nodes, updating values.
        Parameters:
            node: chosen node of maximum value
            allchild_graphs: list of graphs of all the node's children
        '''
        while node != None:
            node.allchild.extend(allchild_graphs)
            node.visits += 1
            npmean, npvar = self.gp.predict(node.allchild, return_cov=True)
            npvar.flat[::len(npvar) + 1] += 1e-10
            npvar_inv = np.linalg.inv(npvar)
            child_mean = (np.sum(npvar_inv@npmean))/np.sum(npvar_inv)
            child_sigma = np.sqrt(1 / np.sum(npvar_inv))
            exploitation = norm(child_mean, child_sigma).pdf(self.target)
            exploration = np.sqrt(np.log(self.tree.visits / node.visits))
            node.post_mean = child_mean
            node.post_var = child_sigma
            node.mlt = exploitation + self.exploration_constant * exploration
            node = node.parent
    
    def solve(self):
        ''' Run MCTS and select best action. 
        Returns:
            found: boolean, whether or not the node has been found
            max_node: TreeNode, maximum node in the tree
        '''
        for i in range(self.iterations):
            max_node, depth = self.selection()
            if depth >= self.depth:
                break
            created = self.expansion(max_node)
            if not created: # No more possible actions can be made
                break
            found, child, allchild_graphs = self.simulation(max_node)
            if found: # Molecule found
                return found, child
            self.backpropagation(max_node, allchild_graphs)
            max_node.selected = True
        max_node = self.tree
        max_mlt = self.tree.mlt
        while len(max_node.children) != 0: 
            prev_max = max_node
            for child in max_node.children: 
                if child.mlt > max_mlt:
                    max_mlt = child.mlt
                    max_node = child
            if max_node == prev_max:
                break
        print("-----------------------------------------------------------")
        print(self.tree)
        return False, max_node

