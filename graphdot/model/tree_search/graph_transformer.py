#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from scipy.stats import norm
from graphdot.util.iterable import argmax
from ._tree import Tree


class MCTSGraphTransformer:
    '''A varient of Monte Carlo tree search for optimization and root-finding
    in a space of graphs.

    Parameters
    ----------
    rewriter: callable
        A callable that implements the :py:class:`Rewriter` abstract class.
    surrogate: object
        A predictor used to calculate the target property of a given graph.
    exploration_bias: float
        Tunes the preference of the MCTS model between exploitation and
        exploration of the search space.
    precision: float
        Target precision of MCTS search outcome.
    '''

    def __init__(self, rewriter, surrogate, exploration_bias=1.0,
                 precision=0.01):
        self.rewriter = rewriter
        self.surrogate = surrogate
        self.exploration_bias = exploration_bias
        self.precision = precision

    def seek(self, g0, target, maxiter=500, return_tree=False,
             random_state=None):
        '''Transforms an initial graph into one with a specific desired target
        property value.

        Parameters
        ----------
        g0: object
            A graph to start the tree search with.
        target: float
            Target property value of the desired graph.
        maxiter: int
            Maximum number of MCTS iterations to perform.
        return_tree: Boolean
            Whether or not to return the search tree in its original form or as
            a flattened dataframe.
        random_state: int or :py:`np.random.Generator`
            The seed to the random number generator (RNG), or the RNG itself.
            If None, the default RNG in numpy will be used.

        Returns
        -------
        tree: DataFrame
            If `return_tree` is True, a hierarchical dataframe representing
            the search tree will be returned; otherwise, a flattened dataframe
            will be returned.
        '''
        random_state = self._parse_random_state(random_state)
        tree = self._spawn(None, [g0])
        self._evaluate(tree)
        for _ in range(maxiter):
            self._mcts_step(
                tree,
                lambda nodes: self._likelihood_ucb(target, nodes),
                random_state=random_state
            )
        if return_tree is True:
            return tree
        else:
            df = tree.flat
            df['likelihood'] = self._likelihood(target, df)
            return df.to_pandas().sort_values(['likelihood'], ascending=False)

    @staticmethod
    def _parse_random_state(random_state):
        if isinstance(random_state, np.random.Generator):
            return random_state
        elif random_state is not None:
            return np.random.Generator(np.random.PCG64(random_state))
        else:
            return np.random.default_rng()

    def _spawn(self, node, leaves):
        return Tree(
            parent=[node] * len(leaves),
            children=[None] * len(leaves),
            g=leaves,
            visits=np.zeros(len(leaves), dtype=np.int)
        )

    def _likelihood(self, target, nodes):
        return norm.pdf(
            target, nodes.tree_mean, np.maximum(nodes.tree_std, self.precision)
            # This line below does not work, especially the '+' part:
            # target, nodes.tree_mean, nodes.tree_std + self.precision
        )

    def _confidence_bounds(self, nodes):
        return self.exploration_bias * np.sqrt(
            np.log(nodes.parent[0].visits) / nodes.visits
        )

    def _likelihood_ucb(self, target, nodes):
        return self._likelihood(target, nodes) + self._confidence_bounds(nodes)

    def _evaluate(self, nodes):
        mean, cov = self.surrogate.predict(nodes.g, return_cov=True)
        nodes['self_mean'] = mean.copy()
        nodes['tree_mean'] = mean.copy()
        nodes['self_std'] = cov.diagonal()**0.5
        nodes['tree_std'] = cov.diagonal()**0.5
        nodes['score'] = np.zeros_like(mean)
        nodes.visits += 1

    def _mcts_step(self, tree, score_fn, random_state):
        '''selection'''
        n = next(tree.iternodes())
        n.visits += 1
        while n.children is not None:
            n = argmax(
                n.children.iternodes(),
                lambda i, j: i.score < j.score
            )
            n.visits += 1

        '''expansion'''
        n.children = self._spawn(n, self.rewriter(n, random_state))

        '''simulate'''
        self._evaluate(n.children)

        '''back-propagate'''
        p = n
        while p:
            p.tree_mean = np.average(
                p.children.tree_mean,
                weights=p.children.tree_std**-2
            )
            p.tree_std = np.average(
                (p.children.tree_mean - p.tree_mean)**2,
                weights=p.children.tree_std**-2
            )**0.5
            p.children['score'] = score_fn(p.children)
            p = p.parent
