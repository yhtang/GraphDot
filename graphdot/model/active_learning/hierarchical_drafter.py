#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np


class HierarchicalDrafter:
    '''Hierarhically select representative samples from a large dataset where
    a direct algorithm can be prohibitively expensive.

    Parameters
    ----------
    selector: callable
        A selection algorithm that can pick a given number of samples from a
        dataset to maximize a certain acquisition function.
    k: int > 1
        The branching factor of the search hierarchy.
    batch: int
        The size of the leaf batches in the search hierarchy.
    '''

    def __init__(self, selector, k=2, batch='auto'):
        assert k > 1, "k must be an integer greater than 1"
        assert callable(selector)
        self.selector = selector
        self.k = k
        self.batch = batch

    def __call__(self, X, n, random_state=None):
        '''Find a n-sample subset of X that attempts to maximize a certain
        diversity criterion.

        Parameters
        ----------
        X: feature matrix or list of objects
            Input dataset.
        n: int
            The size of the subset to be chosen.
        random_state: int or :py:`np.random.Generator`
            The seed to the random number generator (RNG), or the RNG itself.
            If None, the default RNG in numpy will be used.

        Returns
        -------
        chosen: list
            A sorted list of indices of the samples that are chosen.
        '''
        assert len(X) >= n
        if not isinstance(X, np.ndarray):
            X = np.asarray(X, np.object)

        if self.batch == 'auto':
            batch = self.k * n
        else:
            assert self.batch >= n, "Batch size must be greater than n!"
            batch = self.batch

        if isinstance(random_state, np.random.Generator):
            rng = random_state
        elif random_state is not None:
            rng = np.random.Generator(np.random.PCG64(random_state))
        else:
            rng = np.random.default_rng()

        return np.sort(
            self._pick(
                X,
                rng.permutation(len(X)),
                n,
                batch
            )
        )

    def _pick(self, X, nominee, n, batch):
        if len(nominee) > batch:
            '''divide and conquer'''
            stops = np.linspace(0, len(nominee), self.k + 1, dtype=np.int)
            nominee = np.concatenate([
                self._pick(X, nominee[b:e], n, batch)
                for b, e in zip(stops[:-1], stops[1:])
            ])

        return nominee[self.selector(X[nominee], n)]
