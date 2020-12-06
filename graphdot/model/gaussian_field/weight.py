#!/usr/bin/env python
# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
import numpy as np


class Weight(ABC):

    @abstractmethod
    def __call__(self, X, Y=None, eval_gradient=False):
        '''Computes the weight matrix and optionally its gradient with respect
        to hyperparameters.

        Parameters
        ----------
        X: list
            The first dataset to be compared.
        Y: list or None
            The second dataset to be compared. If None, X will be compared with
            itself.
        eval_gradient: bool
            If True, returns the gradient of the weight matrix alongside the
            matrix itself.

        Returns
        -------
        weight_matrix: 2D ndarray
            A weight matrix between the datasets.
        weight_matrix_gradients: 3D ndarray
            A tensor where the i-th frontal slide [:, :, i] contain the partial
            derivative of the weight matrix with respect to the i-th
            hyperparameter.
        '''

    @property
    @abstractmethod
    def theta(self):
        '''An ndarray of all the hyperparameters in log scale.'''

    @theta.setter
    @abstractmethod
    def theta(self, values):
        '''Set the hyperparameters from an array of log-scale values.'''

    @property
    @abstractmethod
    def bounds(self):
        '''The log-scale bounds of the hyperparameters as a 2D array.'''


class RBFOverDistance(Weight):
    '''Set weights by applying an RBF onto a distance matrix.

    Parameters
    ----------
    metric: callable
        An object that implements a distance metric.
    sigma: float
        The log scale hyperparameter for the RBF Kernel.
    sigma_bounds: float
        The bounds for sigma.
    X: list
        Dataset.
    compute_D_from_graphs: bool
        Whether or not to compute the D matrix during initialization.
    '''

    def __init__(self, metric, sigma, sigma_bounds, caching=False):
        self.sigma = sigma
        self.sigma_bounds = sigma_bounds
        self.metric = metric
        self.caching = caching

    def __call__(self, X, Y=None, eval_gradient=False):
        if Y is None:
            if self.caching and hasattr(self, 'dXX'):
                d = self.dXX
            else:
                d = self.metric(X)
                if self.caching:
                    self.dXX = d
        else:
            if self.caching and hasattr(self, 'dXY'):
                d = self.dXY
            else:
                d = self.metric(X, Y)
                if self.caching:
                    self.dXY = d

        w = np.exp(-d**2 * self.sigma**-2)
        if eval_gradient:
            return w, np.array([2 * d**2 * w * self.sigma**-3])
        else:
            return w

    @property
    def theta(self):
        return np.log([self.sigma])

    @theta.setter
    def theta(self, values):
        self.sigma = np.exp(values)[0]

    @property
    def bounds(self):
        return np.log(self.sigma_bounds)
