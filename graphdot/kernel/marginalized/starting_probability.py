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
        '''The log-scale hyperparameters of the starting probability
        distribution as an ndarray.'''
        pass

    @theta.setter
    @abstractmethod
    def theta(self):
        '''Sets the hyperparameters of the starting probability distribution
        from log-scale values in an ndarray.'''
        pass

    @property
    @abstractmethod
    def bounds(self):
        '''The log-scale bounds of the hyperparameters as a 2D array.'''
        pass


class Fixed(StartingProbability):
    '''Assigned all nodes the same starting probability that will remain
    the same during training.

    Parameters
    ----------
    p: float
        The starting probability value.
    '''

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
    '''Assigned all nodes the same starting probability.

    Parameters
    ----------
    p: float
        The starting probability value.
    p_bounds: (float, float)
        The range within which the starting probability can vary during
        training.
    '''
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


class Adhoc(StartingProbability):
    '''Converts any callable that takes in a dataframe and produces a starting
    probability array into a StartingProbability instance. However, note that
    ad-hoc starting probabilities cannot be optimized during training.

    Parameters
    ----------
    f: callable
        Any callable that takes in a dataframe and produces a same-length
        ndarray.
    '''

    def __init__(self, f):
        self.f = f

    def __call__(self, nodes):
        return self.f(nodes), []

    @property
    def theta(self):
        return tuple()

    @theta.setter
    def theta(self, t):
        pass

    @property
    def bounds(self):
        return tuple()
