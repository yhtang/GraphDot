#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from graphdot.util.iterable import argmax


class LikelihoodDrivenGraphRewrite:
    '''A varient of Monte Carlo tree search for finding graphs with desired
    propreties.

    Parameters
    ----------
    target: float
        Desired value for the target property.
    predictor: callable
        A predictor used to calculate the target property of a given graph.
    '''

    def __init__(self, rewriter, evaluator, alpha=1e-8):
        self.rewriter = rewriter
        self.evaluator = evaluator
        self.alpha = alpha

    def step(self, root):
        # selection
        n = self.tree.root
        while n.children is not None:
            n.visits += 1
            n = argmax(
                n.children,
                lambda i, j: self.score(i) < self.score(j)
            )

        # expansion
        n.children = self.rewriter(n)

        # simulate
        n.children.mean, n.children.cov = self.evaluator(n.children)
        n.children.var = n.children.cov.diagonal()
        n.children.cov.flat[::len(n.children.cov) + 1] *= 1 + self.alpha

        cov_inv = np.linalg.inv(n.children.cov)
        n.mean = np.sum(cov_inv @ n.children.mean) / np.sum(cov_inv)
        n.var = 1 / np.sum(cov_inv)

        # back-propagate
        p = n.parent
        while p:
            c = np.copy(p.children.cov)
            c.flat[::len(c) + 1] += p.children.var
            c_inv = np.linalg.inv(c)
            p.mean = np.sum(c_inv @ p.children.mean) / np.sum(c_inv)
            p.var = 1 / np.sum(c_inv)
