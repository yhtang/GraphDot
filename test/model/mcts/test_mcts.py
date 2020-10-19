#!/usr/bin/env python
# -*- coding: utf-8 -*-

from mcts import Rewrite
from scipy.stats import norm


class RewriteString(Rewrite):
    ''' Rewrite operation for string. 
    Each graph is of the form 'A'*n + 'B'*n and the target graph we want has 3 A's and 3 B's.
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
    def __init__(self):
        pass
    def predict(self, graphs):
        mean = np.array([(self.add(graph)) // 2 for graph in graphs])
        covariance = np.eye(len(graphs))
        for i in range(len(graphs)):
            for j in range(i+1, len(graphs)):
                cov = abs(graphs[i].count("A") - graphs[j].count("A")) + abs(graphs[i].count("B") - graphs[j].count("B"))
                covariance[i][j], covariance[j][i] = cov, cov
        return mean, covariance