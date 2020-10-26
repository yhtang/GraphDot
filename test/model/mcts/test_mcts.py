#!/usr/bin/env python
# -*- coding: utf-8 -*-

from mcts import Rewrite
from scipy.stats import norm


class RewriteString(Rewrite):
    ''' Rewrite operation for string. 
    Each graph is of the form 'A'*n + 'B'*n and the target graph we want has 3 A's.
    Each rewrite operation either adds an 'A' or 'B', replaces an 'A' with a 'B', or deletes a letter (if len(graph) != 0)
    '''
    def __init__(self):
        super(__init__)
    def __call__(self, graph, random_state):
        ''' Returns a newly rewritten string. 
        '''
        np.random.seed(random_state)
        possible_actions = [self._add(graph),self._substitute(graph),self._delete(graph)][np.random.randint(2)]
        action = np.array(possible_actions)[np.random.choice(len(possible_actions), min(len(possible_actions),self.width), False)]
        return action
    def _add(self, original):
        ''' Returns all possible strings after having inserted a new letter into the original string.
        '''
        result = set()
        for i in range(len(original)):
            result.add(original[:i] + "A" + original[i:])
            result.add(original[:i] + "B" + original[i:])
        return result
    def _substitute(self, original):
        '''  Returns all possible strings after having substituted one letter with another in the original string.
        '''
        reverse = {"A": "B", "B": "A"}
        result = set()
        for i in range(len(original)):
            result.add(original[:i] + reverse[original[i]] + original[i+1:])
        return result
    def _delete(self, original):
        '''  Returns all possible strings after having deleted a letter from the original string.
        '''
        result = set()
        for i in range(len(original)):
            result.add(original[:i] + original[i+1:])
        return result

class PredictorString():
    '''  Returns number of 'A's in a string.
    '''
    def __init__(self):
        pass
    def predict(self, graphs):
        mean = np.array([(graph.count("A") for graph in graphs])
        covariance = np.eye(len(graphs))
        for i in range(len(graphs)):
            for j in range(i+1, len(graphs)):
                cov = abs(graphs[i].count("A") - graphs[j].count("A")) + abs(graphs[i].count("B") - graphs[j].count("B"))
                covariance[i][j], covariance[j][i] = cov, cov
        return mean, covariance

def test_newnode(string):
    print("Input String: ", string)
    predictor = PredictorString()
    prediction, sigma = predictor.predict(string)
    found = False
    while not found:
        mcts = MCTS(predictor = predictor, seed_graph=string, tree=TreeNode(children=[], parent=None, graph=string, visits=1, allchild=[], depth=0), width = 3, iterations=5, depth=5, target = 3, margin = 0, sigma_margin = 0, exploration_constant = 1)
        found, string = mcts.solve()
        newPrediction = predictor.predict(string.graph)
        assert (target - prediction) >= (target - newPrediction) # New node is better than old
        prediction = newPrediction
    return string

def test_checkfound(string, target=3, margin=1, sigma_margin=0.1):
    print("Input String: ", string)
    predictor = PredictorString()
    prediction, sigma = predictor.predict(string)
    mcts = MCTS(predictor = predictor, seed_graph=string, tree=TreeNode(children=[], parent=None, graph=string, visits=1, allchild=[], depth=0), width = 3, iterations=5, depth=5, target = target, margin = margin, sigma_margin = sigma_margin, exploration_constant = 1)
    found = mcts.check_found(self, mcts.tree, prediction, sigma)
    assert not found
    found = mcts.check_found(self, TreeNode(children=[], parent=None, graph="ABAA"), 0.01)
    assert found
    found = mcts.check_found(self, TreeNode(children=[], parent=None, graph="ABA"), 0.01)
    assert found
    found = mcts.check_found(self, TreeNode(children=[], parent=None, graph="BAAA"), 0.2)
    assert not found
    found = mcts.check_found(self, TreeNode(children=[], parent=None, graph="BBBBBA"), 0.05)
    assert not found

def test_selection(string):
    print("Input String: ", string)
    predictor = PredictorString()
    #       root
    #     /      \
    #  childA  childB
    #    |
    # childC
    root = TreeNode(children=[], parent=None, graph=string, visits=1, allchild=[], depth=0)
    childA = TreeNode(children=[], parent=root, graph=string, visits=1, allchild=[], depth=0)
    childB = TreeNode(children=[], parent=root, graph=string, visits=1, allchild=[], depth=0)
    childC = TreeNode(children=[], parent=childA, graph=string, visits=1, allchild=[], depth=0)
    childA.children = [childC]
    root.children = [childA, childB]
    root.score = 3
    childA.score = 4
    childB.score = 2
    childC.score = 6
    mcts = MCTS(predictor = predictor, seed_graph=string, tree=root, width = 3, iterations=5, depth=5, target = 3, margin = 0, sigma_margin = 0, exploration_constant = 1)
    max_node, depth = mcts.selection()
    assert max_node is childC
    assert depth == 2
    childA.score = 2
    childB.score = 4
    max_node, depth = mcts.selection()
    assert max_node is childB
    assert depth == 1

def test_expansion(string):
    print("Input String: ", string)
    predictor = PredictorString()
    #       root
    #     /      \
    #  childA  childB
    #    |
    # childC
    root = TreeNode(children=[], parent=None, graph=string, visits=1, allchild=[], depth=0)
    childA = TreeNode(children=[], parent=root, graph=string, visits=1, allchild=[], depth=0)
    childB = TreeNode(children=[], parent=root, graph=string, visits=1, allchild=[], depth=0)
    childC = TreeNode(children=[], parent=childA, graph=string, visits=1, allchild=[], depth=0)
    childA.children = [childC]
    root.children = [childA, childB]
    mcts = MCTS(predictor = predictor, seed_graph=string, tree=root, width = 3, iterations=5, depth=5, target = 3, margin = 0, sigma_margin = 0, exploration_constant = 1)
    children = mcts.expansion(root)
    assert len(mcts.tree.children) > 2
    assert mcts.tree.children == children
    children = mcts.expansion(childC)
    assert childC.children is not None

def test_simulation(string):
    print("Input String: ", string)
    predictor = PredictorString()
    #       root
    #     /      \
    #  childA  childB
    #    |
    # childC
    root = TreeNode(children=[], parent=None, graph=string, visits=1, allchild=[], depth=0)
    mcts = MCTS(predictor = predictor, seed_graph=string, tree=root, width = 3, iterations=5, depth=5, target = 3, margin = 0, sigma_margin = 0, exploration_constant = 1)
    children = mcts.expansion(root)
    result, child, graphs = mcts.selection(root)
    assert mcts.tree.children is not None
    if result:
        assert child is not None
    else:
        assert child is None
    for child in root.children:
        assert child.score is not None

def test_backpropagation(string):
    print("Input String: ", string)
    predictor = PredictorString()
    #       root
    #     /      \
    #  childA  childB
    #    |
    # childC
    root = TreeNode(children=[], parent=None, graph=string, visits=1, allchild=[], depth=0)
    childA = TreeNode(children=[], parent=root, graph=string, visits=1, allchild=[], depth=0)
    childB = TreeNode(children=[], parent=root, graph=string, visits=1, allchild=[], depth=0)
    childC = TreeNode(children=[], parent=childA, graph=string, visits=1, allchild=[], depth=0)
    childA.children = [childC]
    childA.allchild = [childC]
    root.children = [childA, childB]
    root.allchild = [childA, childB, childC]
    root.score = 1
    childA.score = 2
    mcts = MCTS(predictor = predictor, seed_graph=string, tree=root, width = 3, iterations=5, depth=5, target = 3, margin = 0, sigma_margin = 0, exploration_constant = 1)
    children = mcts.backpropagation(childA)
    assert childA.score != 1
    assert root.score != 1
    assert childA.post_mean is not None
    assert root.post_mean is not None
    assert childA.post_var is not None
    assert root.post_var is not None