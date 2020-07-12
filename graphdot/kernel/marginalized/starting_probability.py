#!/usr/bin/env python
# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from collections import namedtuple
import numpy as np


class StartingProbability(ABC):
    '''Assigns non-negative starting probabilities to each node of a graph.
    Note that such a notion of starting probability can be safely generalize so
    that the probabilies does not have to sum to 1.
    '''

    @abstractmethod
    def __call__(self, nodes):
        '''Takes in a dataframe of nodes and returns an array of probabilities.

        Parameters
        ----------
        nodes: DataFrame
            Each node corresponds to a row in the data frame.

        Returns
        -------
        p: numpy.ndarray
            The starting probabilities on each node.
        d_p: numpy.ndarray
            The gradient of the starting probabilities with respect to the
            hyperparameters as a matrix where each row corresponds to one
            hyperparameter.
        '''
        pass

    @property
    @abstractmethod
    def theta(self):
        pass

    @theta.setter
    @abstractmethod
    def theta(self):
        pass

    @property
    @abstractmethod
    def bounds(self):
        pass


class Fixed(StartingProbability):

    def __init__(self, p):
        self.p = p

    def __call__(self, nodes):
        return self.p * np.ones(len(nodes)), np.ones((1, len(nodes)))

    @property
    def theta(self):
        return namedtuple('StartingProbability', ['p'])(
            self.p
        )

    @theta.setter
    def theta(self, t):
        self.p = t[0]

    @property
    def bounds(self):
        return ((self.p, self.p),)


class Uniform(StartingProbability):

    def __init__(self, p, p_bounds=(1e-3, 1e3)):
        assert(isinstance(p_bounds, tuple) and len(p_bounds) == 2)
        self.p = p
        self.p_bounds = p_bounds

    def __call__(self, nodes):
        return self.p * np.ones(len(nodes)), np.ones((1, len(nodes)))

    @property
    def theta(self):
        return namedtuple('StartingProbability', ['p'])(
            self.p
        )

    @theta.setter
    def theta(self, t):
        self.p = t[0]

    @property
    def bounds(self):
        return (self.p_bounds,)
