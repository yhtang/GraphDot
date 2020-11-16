#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
# from .tree import Tree
# from graphdot.minipandas import DataFrame
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

    def __init__(self, rewriter, evaluator, alpha=1e-8):
        self.rewriter = rewriter
        self.evaluator = evaluator
        self.alpha = alpha

    def __call__(self, seed, target):
        tree = Tree(
            parent=[None],
            children=[None],
            label=[seed],
            visits=np.zeros(1)
        )
        self.evaluate(tree)
        # while True:
        for _ in range(2):
            self.step(next(tree.iternodes()))

        print(f'Tree\n{str(tree)}')

    def score(self, node):
        return True

    def evaluate(self, nodes):
        mean, cov = self.evaluator.predict(nodes.label, return_cov=True)
        cov.flat[::len(cov) + 1] += self.alpha
        nodes['mean'] = mean
        nodes['cov'] = cov
        nodes['uncertainty'] = cov.diagonal()**0.5
        nodes.visits += 1

    def step(self, root):
        # selection
        n = root
        while n.children is not None:
            n.visits += 1
            n = argmax(
                n.children.iternodes(),
                lambda i, j: self.score(i) < self.score(j)
            )

        # expansion
        child_graphs = self.rewriter(n)
        print(f'type(n) {type(n)}')
        n.children = Tree(
            parent=[n] * len(child_graphs),
            children=[None] * len(child_graphs),
            label=child_graphs,
            visits=np.zeros(len(child_graphs))
        )

        # simulate
        self.evaluate(n.children)

        # back-propagate
        p = n
        while p:
            c = np.copy(p.children.cov)
            c.flat[::len(c) + 1] += p.children.uncertainty**2
            c_inv = np.linalg.inv(c)
            p.mean = np.sum(c_inv @ p.children.mean) / np.sum(c_inv)
            p.uncertainty = np.sqrt(1 / np.sum(c_inv))
            p = p.parent
