#!/usr/bin/env python
# -*- coding: utf-8 -*-
from graphdot.kernel.marginalized import MarginalizedGraphKernel


class MaxiMin(MarginalizedGraphKernel):
    '''The maximin distance metric, also known as the supinf distance or the
    Hausdorff distance named after Felix Hausdorff, measures how far two point
    sets are from each other. Informally, the maximin distance is the greatest
    of all the distances from a point in one set to the closest point in the
    other set. Two sets are close in the maximin distance if every point of
    either set is close to some point of the other set.

    Parameters
    ----------
    args, kwargs: 
        Inherits :py:class:`MarginalizedGraphKernel`.
    '''

    def __init__(self, *args, **kwargs):
        self.kernel = kernel
        self.kernel_options = kernel_options

    def __call__(self, X, Y=None, eval_gradient=False, **options):
        '''Computes the distance matrix and optionally its gradient with respect
        to hyperparameters.

        Parameters
        ----------
        X: list of graphs
            The first dataset to be compared.
        Y: list of graphs or None
            The second dataset to be compared. If None, X will be compared with
            itself.
        eval_gradient: bool
            If True, returns the gradient of the weight matrix alongside the
            matrix itself.
        options: keyword arguments
            Additional arguments to be passed to the underlying kernel.

        Returns
        -------
        M: 2D ndarray
            A distance matrix between the data points.
        dM: 3D ndarray
            A tensor where the i-th frontal slide [:, :, i] contain the partial
            derivative of the distance matrix with respect to the i-th
            hyperparameter.
        '''

    @property
    def theta(self):
        '''An ndarray of all the hyperparameters in log scale.'''
        return self.kernel.theta

    @theta.setter
    def theta(self, values):
        '''Set the hyperparameters from an array of log-scale values.'''
        self.kernel.theta = values

    @property
    def bounds(self):
        '''The log-scale bounds of the hyperparameters as a 2D array.'''
        return self.kernel.bounds
