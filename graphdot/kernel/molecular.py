#!/usr/bin/env python
# -*- coding: utf-8 -*-
from graphdot.kernel.marginalized import MarginalizedGraphKernel
from graphdot.kernel.marginalized.basekernel import KroneckerDelta
from graphdot.kernel.marginalized.basekernel import SquareExponential
from graphdot.kernel.marginalized.basekernel import TensorProduct


class Tang2019MolecularKernel:
    """A margianlized graph kernel for **3D molecular structures** as in:
    Tang, Y. H., & de Jong, W. A. (2019). Prediction of atomization energy
    using graph kernel and active learning. *The Journal of chemical physics*,
    150(4), 044107.
    The kernel can be directly used together with Graph.from_ase() to operate
    on molecular structures.

    Parameters
    ----------
    stopping_probability: float in (0, 1)
        The probability for the random walk to stop during each step.
    element_prior: float in (0, 1)
        The baseline similarity between distinct elements --- an element
        always have a similarity 1 to itself.
    edge_length_scale: float in (0, inf)
        length scale of the Gaussian kernel on edge length. A rule of thumb is
        that the similarity decays smoothly from 1 to nearly 0 around three
        times of the length scale.
    """

    def __init__(self, stopping_probability=0.01, element_prior=0.2,
                 edge_length_scale=0.05):
        self.stopping_probability = stopping_probability
        self.element_prior = element_prior
        self.edge_length_scale = edge_length_scale
        self._makekernel()

    def _makekernel(self):
        self.kernel = MarginalizedGraphKernel(
            TensorProduct(element=KroneckerDelta(self.element_prior, 1.0)),
            TensorProduct(length=SquareExponential(self.edge_length_scale)),
            q=self.stopping_probability
        )

    def __call__(self, X, Y=None):
        return self.kernel(X, Y)