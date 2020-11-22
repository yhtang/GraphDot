#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from scipy.stats import norm
from graphdot.util.iterable import argmax
from .tree import Tree


class LikelihoodDrivenTreeSearch:
    '''A varient of Monte Carlo tree search for finding graphs with desired
    propreties.

    Parameters
    ----------
    target: float
        Desired value for the target property.
    predictor: callable
        A predictor used to calculate the target property of a given graph.
    '''

    def __init__(self, rewriter, evaluator, exploration_bias=1.0, alpha=1e-10):
        self.rewriter = rewriter
        self.evaluator = evaluator
        self.exploration_bias = exploration_bias
        self.alpha = alpha

    def search(self, seed, target):
        tree = self._spawn(None, [seed])
        self._evaluate(tree)
        for _ in range(20):
            print(f'{tree}\n\n')
            self._step(
                tree,
                lambda nodes: self._log_likelihood_ucb(target, nodes)
            )
        print(f'{tree}\n\n')
        return tree

    def _spawn(self, node, leaves):
        return Tree(
            parent=[node] * len(leaves),
            children=[None] * len(leaves),
            state=leaves,
            visits=np.zeros(len(leaves), dtype=np.int)
        )

    def _log_likelihood_ucb(self, target, nodes):
        return (
            norm.pdf(target, nodes.tree_mean, nodes.tree_std)
            + self.exploration_bias * np.sqrt(
                np.log(nodes.parent[0].visits) / nodes.visits
            )
        )

    def _evaluate(self, nodes):
        mean, cov = self.evaluator.predict(nodes.state, return_cov=True)
        nodes['self_mean'] = mean.copy()
        nodes['self_std'] = cov.diagonal()**0.5
        nodes['tree_mean'] = mean.copy()
        nodes['tree_std'] = cov.diagonal()**0.5
        nodes['score'] = np.zeros_like(mean)
        nodes.visits += 1

    def _step(self, tree, score_fn):
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
