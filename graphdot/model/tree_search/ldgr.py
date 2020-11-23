#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from scipy.stats import norm
from graphdot.util.iterable import argmax
from .tree import Tree


class MCTSGraphTransformer:
    '''A varient of Monte Carlo tree search for optimization in a space of
    graphs.

    Parameters
    ----------
    predictor: callable
        A predictor used to calculate the target property of a given graph.
    '''

    def __init__(self, rewriter, surrogate, exploration_bias=1.0,
                 precision=0.01):
        self.rewriter = rewriter
        self.surrogate = surrogate
        self.exploration_bias = exploration_bias
        self.precision = precision

    def seek(self, g0, target, maxiter=500, return_tree=False):
        tree = self._spawn(None, [g0])
        self._evaluate(tree)
        for _ in range(maxiter):
            self._mcts_step(
                tree,
                lambda nodes: self._likelihood_ucb(target, nodes)
            )
        if return_tree is True:
            return tree
        else:
            df = tree.flat
            df['likelihood'] = norm.pdf(
                target, df.self_mean, np.maximum(self.precision, df.self_std)
            )
            return df.to_pandas().sort_values(['likelihood'], ascending=False)

    def _spawn(self, node, leaves):
        return Tree(
            parent=[node] * len(leaves),
            children=[None] * len(leaves),
            g=leaves,
            visits=np.zeros(len(leaves), dtype=np.int)
        )

    def _likelihood(self, target, nodes):
        return norm.pdf(target, nodes.tree_mean,
                        np.maximum(self.precision, nodes.tree_std))

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

    def _mcts_step(self, tree, score_fn):
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
        n.children = self._spawn(n, self.rewriter(n))

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


# def monte_carlo_tree_search(f, xin, xgen, exploration_bias=1.0,
#                             precision=0.01, maxiter=500, return_tree=False):
#     mcts = MonteCarloTreeSearch(
#         rewriter=xgen,
#         surrogate=lambda x, return_std: (f(x), np.zeros(len(x))),
#         exploration_bias=exploration_bias,
#         precision=precision
#     )
#     tree = mcts.find(xin, 0, maxiter=maxiter)
#     if return_tree is True:
#         return tree
#     else:
#         df = tree.flat.to_pandas()
#         df['likelihood'] = norm.pdf(0, df.self_mean,
#                                     np.maximum(precision, df.self_std))
#         return df.sort_values(['likelihood'], ascending=False)
