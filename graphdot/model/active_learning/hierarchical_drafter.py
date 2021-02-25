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
    a: float in (1, k]
        The multiplier to the number of samples that each level need to
        generate during hierarchical screening. For example, if n samples are
        wanted in the end, then the immediate next level should forward
        at least m * n samples for the last level drafter to choose from.
    leaf_ratio: float in (0, 1)
        If ratio berween output and input samples is greater than it, stop
        further division and carry out selection using the given selector.
    '''

    def __init__(self, selector, k=2, a=2, leaf_ratio='auto'):
        assert k > 1, "k must be an integer greater than 1"
        assert callable(selector)
        self.selector = selector
        self.k = k
        self.a = a
        if leaf_ratio == 'auto':
            self.leaf_ratio = 0.5
        else:
            self.leaf_ratio = leaf_ratio

    def __call__(self, X, n, random_state=None, verbose=False):
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

        if isinstance(random_state, np.random.Generator):
            rng = random_state
        elif random_state is not None:
            rng = np.random.Generator(np.random.PCG64(random_state))
        else:
            rng = np.random.default_rng()

        return np.sort(self._pick(X, rng.permutation(len(X)), n,
                                  verbose=verbose))

    def _pick(self, X, nominee, n, lvl=0, verbose=False):
        if verbose:
            print(
                (' ' * lvl) + f'C_{len(nominee)}_{n}',
                n / len(nominee),
                self.leaf_ratio
            )
        if len(nominee) <= n:
            return nominee
        elif n / len(nominee) < self.leaf_ratio and n > self.k / self.a:
            '''divide and conquer'''
            stops = np.linspace(0, len(nominee), self.k + 1, dtype=np.int)
            nominee = np.concatenate([
                self._pick(
                    X, nominee[b:e], int(n * self.a // self.k), lvl + 1,
                    verbose=verbose
                )
                for b, e in zip(stops[:-1], stops[1:])
            ])
        return nominee[self.selector(X[nominee], n)]
