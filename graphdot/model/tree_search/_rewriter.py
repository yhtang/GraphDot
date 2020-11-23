#!/usr/bin/env python
# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod


class AbstractRewriter(ABC):
    ''' Abstract base class for graph rewrite rules. '''

    @abstractmethod
    def __call__(self, g, random_state):
        ''' Rewrite the given graph using a rule drawn randomly from a pool.

        Parameters
        ----------
        g: object
            An input graph to be transformed.
        random_state: int or :py:`np.random.Generator`
            The seed to the random number generator (RNG), or the RNG itself.
            If None, the default RNG in numpy will be used.

        Returns
        -------
        H: list
            A list of new graphs transformed from `g`.
        '''


class SMILESRewriter(AbstractRewriter):
    pass
