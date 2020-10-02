#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import random
from abc import ABC
from scipy.stats import norm
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

class Rewrite(ABC):
    ''' Abstract base class for Rewrite operations. '''

    @abstractmethod
    @classmethod
    def add(args):
        ''' Adds a new node to the graph.
        Parameters
        ----------
        ???
        Returns
        -------
        added_graph: graph
            Returns the new graph that is produced, or None if not possible. 
        '''
        return None

    @abstractmethod
    @classmethod
    def substitute(original_graph, subgraph, newgraph):
        ''' Substitutes a subgraph with another subgraph.

        Parameters
        ----------
        original_graph: graph
            The full graph being rewritten.
        subgraph: graph
            The subgraph being taken out.
        newgraph: graph
            The new subgraph being substituted in.
            
        Returns
        -------
        graph_lst: list
            Returns the list of graphs that can be produced with this operation, and None if not possible.
            Note that more than one graph can be produced because the given subgraph may occur in multiple
            parts of the original graph. 
        '''
        return None
    
    @abstractmethod
    @classmethod
    def delete(smiles, substruct):
        ''' Deletes the given subgraph from the original graph.

        Parameters
        ----------
        original_graph: graph
            The full graph being rewritten.
        subgraph: graph
            The subgraph being taken out.

        Returns
        ------
        graph_lst: list
            Returns the list of graphs that can be produced with this operation, and None if not possible.
            More than one graph can be produced because the given subgraph may occur in multiple
            parts of the original graph. 
        '''
        return None

class TreeNode:
    ''' A MCTS tree node.

    Parameters
    -----------
    children: list
        A list of TreeNodes of the current Node's children
    parent: TreeNode
        The current Node's parent. If the current Node is the root node, parent is None.
    graph: 
        String representation of the graph being represented.
    visits: integer
        The number of times this node has been visited.
    allchild: list of Graphs
        The list of all graphs in the current node's subtree.
    depth: integer
        The depth of the tree.
    '''

    def __init__(self, children, parent, graph, visits, allchild, depth):
        self.children = children
        self.parent = parent
        self.graph = graph
        self.visits = visits
        self.allchild = allchild
        self.mean = 0
        self.std = 0
        self.post_mean = 0
        self.post_var = 0
        self.score = 0
        self.depth = depth
        self.selected = False

    def __repr__(self):
        if self.children:
            branch_str = ', ' + repr(self.children)
        else:
            branch_str = ''
        return 'Node({0}{1})'.format(self.graph, branch_str)

    def __str__(self):
        def print_tree(t, indent=0):
            space = 50 - (t.depth*4 + len(t.graph))
            if t.selected: select = 1
            else: select = 0
            if t.children:
                tree_str = '    ' * indent + '*' * select + t.graph + " " * space + "(Post_mean: " + str(t.post_mean)[:5] + " Post_var: " + str(t.post_var)[:5] + " score: " + str(t.score)[:5] + ")\n"
                for child in t.children:
                    tree_str += print_tree(child, indent + 1)
            else:
                tree_str = '    ' * indent + '*' * select + t.graph + " " * space + "(Mean: " + str(t.mean)[:5] + " Std: " + str(t.std)[:5] + " score: " + str(t.score)[:5] + ")\n"
            return tree_str
        return print_tree(self).rstrip()

class MCTS():
    ''' Monte Carlo Tree Search algorithm. 
    Parameters
    ----------
    gp: Predictor instance
        A predictor used to calculate the corresponding value of a given graph. 
        This predictor should be have the predict() function defined, which returns
        the mean and covariance matrix of the prediction.
    seed_graph: graph
        The seed graph to start the MCTS algorithm from.
    tree: TreeNode instance
        The state of the MCTS tree at the beginning of the MCTS iteration.
    width: integer
        The number of maximum nodes allowed in one level.
    iterations: integer
        The number of MCTS iterations to run in each turn.
    depth: integer
        The maximum depth allowed in node selection.
    target: float
        The target value to achieve.
    margin: float
        The margin of error around the target value that can be accepted. 
    sigma_margin: float
        The largest degree of uncertainty that can be accepted.
    exploration_constant: float
        The degree of exploration; this is a hyperparameter to be tuned.
    tograph: function
        A function that
    scoring: string, default = "uct"
        Either "uct" or "score". Determines the scoring function used.
    '''
    
    def __init__(self, gp, seed_graph, tree, width, iterations, depth, target, margin, sigma_margin, exploration_constant, tograph, scoring='uct'):
        self.seed_graph = seed_graph
        self.tree = tree
        self.iterations = iterations
        self.depth = depth
        self.width = width
        self.target = target
        self.gp = gp
        self.margin = margin
        self.sigma_margin = sigma_margin
        self.exploration_constant = exploration_constant
        if scoring != "uct" or scoring != "mlt":
            raise ValueError("Invalid scoring argument")
        self.scoring = scoring
    
    def __str__(self):
        print(self.tree)
    
    def check_found(self, child, pred, sigma):
        ''' Checks to see if a node that fits the given constraints has been found.
        
        Parameters
        ----------
        child: TreeNode
            The input node that is being checked.
        pred: float
            Prediction of the node's associated value.
        sigma: float
            The node's associated uncertainty value.
        
        Returns
        -------
        found: boolean
            Boolean representing whether or not the node has an associated value
            within the acceptable margin of error.
        '''
        if sigma < self.sigma_margin and abs(pred - self.target) < self.margin:
            child.selected = True
            return True
        return False
    
    def selection(self): 
        ''' Select leaf node with maximum score.. 

        Returns
        -------
        max_node: TreeNode
            The node with maximum score.
        depth: integer
            The depth of the max_node found from the root node
        '''
        max_node = self.tree
        max_score = self.tree.score
        depth = 0
        while len(max_node.children) != 0: 
            prev_max = max_node
            depth += 1
            for child in max_node.children: 
                if child.score > max_score:
                    max_score = child.score
                    max_node = child
            if max_node == prev_max:
                children = np.array(max_node.children)
                max_node = children[np.random.randint(0, len(children), 1)[0]]
        return max_node, depth
    
    def expansion(self, node):
        ''' Create all possible outcomes from leaf node. 
        Parameters
        ----------
        node: TreeNode
            The node with the maximum score found by selection()

        Returns
        -------
        created: boolean
            Whether or not new children were added to the given node.
        '''
        graph = node.graph
        addlst, sublst, remlst = Rewrites.getValidActions(graph)
        possible_actions = addlst + sublst + remlst
        possible_actions = np.array(possible_actions)[np.random.choice(len(possible_actions), min(len(possible_actions),self.width), False)]
        for action in possible_actions: 
            try:
                new_graph = action[0]
                child = TreeNode(children=[], parent=node, graph=new_graph, visits=0, allchild=[], depth=node.depth+1)
                node.children.append(child)
            except:
                pass
        return node.children
                
    def simulation(self, node):
        ''' Simulate game from child node's state until it reaches the resulting state of the game. 

        Parameters
        ----------
        node: TreeNode
            The child node found by expansion()

        Returns
        -------
        result: boolean, 
            Whether or not a new node was found.
        child: TreeNode
            The newly found node.
        graphs: list
            The list of graphs of all the node's children
        '''
        graph = node.graph
        graphs = [Graph.from_rdkit(Chem.MolFromSmiles(child.graph)) for child in node.children]
        mean, cov = self.gp.predict(graphs, return_cov=True)
        for i in range(len(node.children)):
            child = node.children[i]
            if self.scoring == "mlt":
                child.mean = mean[i]
                child.std = np.sqrt(cov[i][i])
                exploitation = norm(child.mean, child.std).pdf(self.target)
                exploration = np.sqrt(self.tree.visits)
                child.score = exploitation + self.exploration_constant * exploration
            else:
                # TODO: UCT implementation
                child.score = 0
            child.visits += 1
            if self.check_found(child, child.mean, child.std):
                return True, child, graphs
        return False, None, graphs
    
    def backpropagation(self, node, allchild_graphs):
        ''' Backpropagate against the nodes, updating values.

        Parameters
        ----------
        node: TreeNode
            The chosen node of maximum value
        allchild_graphs: list
            The list of graphs of all the given node's children
        '''
        while node != None:
            node.allchild.extend(allchild_graphs)
            node.visits += 1
            npmean, npvar = self.gp.predict(node.allchild, return_cov=True)
            if self.scoring == "mlt":
                npvar.flat[::len(npvar) + 1] += 1e-10
                npvar_inv = np.linalg.inv(npvar)
                child_mean = (np.sum(npvar_inv@npmean))/np.sum(npvar_inv)
                child_sigma = np.sqrt(1 / np.sum(npvar_inv))
                exploitation = norm(child_mean, child_sigma).pdf(self.target)
                exploration = np.sqrt(np.log(self.tree.visits / node.visits))
                node.post_mean = child_mean
                node.post_var = child_sigma
                node.score = exploitation + self.exploration_constant * exploration
            else:
                # TODO: implement UCT
            node = node.parent
    
    def solve(self):
        ''' Run MCTS and select best action. 

        Returns
        -------
        found: boolean
            Whether or not a node fitting the given constraints has been found.
        max_node: TreeNode
            The node with the maximum score in the tree.
        '''
        for i in range(self.iterations):
            max_node, depth = self.selection()
            if depth >= self.depth:
                break
            created = self.expansion(max_node)
            if not created: # No more possible actions can be made
                break
            found, child, allchild_graphs = self.simulation(max_node)
            if found: # Node found
                return found, child
            self.backpropagation(max_node, allchild_graphs)
            max_node.selected = True
        max_node = self.tree
        max_score = self.tree.score
        while len(max_node.children) != 0: 
            prev_max = max_node
            for child in max_node.children: 
                if child.score > max_score:
                    max_score = child.score
                    max_node = child
            if max_node == prev_max:
                break
        return False, max_node

