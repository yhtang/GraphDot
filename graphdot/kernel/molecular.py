#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
from graphdot.kernel.marginalized import MarginalizedGraphKernel
from graphdot.microkernel import (
    KroneckerDelta,
    SquareExponential,
    TensorProduct
)


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
    starting_probability: float
        The probability for the random walk to start from any node. See the `p`
        kwarg of :class:`graphdot.kernel.marginalized.MarginalizedGraphKernel`
    element_prior: float in (0, 1)
        The baseline similarity between distinct elements --- an element
        always have a similarity 1 to itself.
    edge_length_scale: float in (0, inf)
        length scale of the Gaussian kernel on edge length. A rule of thumb is
        that the similarity decays smoothly from 1 to nearly 0 around three
        times of the length scale.
    """

    def __init__(self,
                 stopping_probability=0.01,
                 starting_probability=1.0,
                 element_prior=0.2,
                 edge_length_scale=0.05, **kwargs):
        self.stopping_probability = stopping_probability
        self.starting_probability = starting_probability
        self.element_prior = element_prior
        self.edge_length_scale = edge_length_scale
        self._makekernel(**kwargs)

    def _makekernel(self, **kwargs):
        self.kernel = MarginalizedGraphKernel(
            TensorProduct(element=KroneckerDelta(self.element_prior)),
            TensorProduct(length=SquareExponential(self.edge_length_scale)),
            q=self.stopping_probability,
            p=self.starting_probability,
            **kwargs
        )

    def __call__(self, X, Y=None, **kwargs):
        """Same call signature as
        :py:meth:`graphdot.kernel.marginalized.MarginalizedGraphKernel.__call__`
        """
        return self.kernel(X, Y, **kwargs)

    def diag(self, X, **kwargs):
        """Same call signature as
        :py:meth:`graphdot.kernel.marginalized.MarginalizedGraphKernel.diag`
        """
        return self.kernel.diag(X, **kwargs)

    @property
    def hyperparameters(self):
        return self.kernel.hyperparameters

    @property
    def theta(self):
        return self.kernel.theta

    @theta.setter
    def theta(self, value):
        self.kernel.theta = value

    @property
    def hyperparameter_bounds(self):
        return self.kernel.hyperparameter_bounds

    @property
    def bounds(self):
        return self.kernel.bounds

    def clone_with_theta(self, theta):
        clone = copy.deepcopy(self)
        clone.theta = theta
        return clone
