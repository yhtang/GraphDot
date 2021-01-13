#!/usr/bin/env python
# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
import copy
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

    def clone_with_theta(self, theta):
        clone = copy.deepcopy(self)
        clone.theta = theta
        return clone


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
    sticky_cache: bool
        Whether or not to save the distance matrix upon first evaluation of the
        weights. This could speedup hyperparameter optimization if the
        underlying distance matrix remains unchanged during the process.
    '''

    def __init__(self, metric, sigma, sigma_bounds=(1e-3, 1e3), mopts={}):
        self.sigma = sigma
        self.sigma_bounds = sigma_bounds
        self.metric = metric
        self.mopts = mopts

    def __call__(self, X, Y=None, eval_gradient=False):
        '''
        Parameters
        ----------
        eval_gradient: bool
            If true, also return the gradient of the weights with respect to
            the **log-scale** hyperparameters.
        '''
        if Y is None:
            Z = (X,)
        else:
            Z = (X, Y)

        if eval_gradient is True:
            D, dD = self.metric(*Z, eval_gradient=True, **self.mopts)
        else:
            D = self.metric(*Z, **self.mopts)

        W = np.exp(-0.5 * D**2 * self.sigma**-2)
        if Y is None:
            W[np.diag_indices_from(W)] = 0
        if eval_gradient:
            dsigma = D**2 * W * self.sigma**-3
            dtheta = (-D * W * self.sigma**-2)[:, :, None] * dD
            dW = np.concatenate(
                [dsigma.reshape(*dsigma.shape, 1), dtheta], axis=2
            )
            return W, dW
        else:
            return W

    @property
    def theta(self):
        return np.concatenate((np.log([self.sigma]), self.metric.theta))

    @theta.setter
    def theta(self, values):
        self.sigma = np.exp(values[0])
        self.metric.theta = values[1:]

    @property
    def bounds(self):
        return np.vstack((
            np.log([self.sigma_bounds]),
            self.metric.bounds
        ))


class RBFOverFixedDistance(Weight):
    '''Set weights by applying an (optimizable) RBF onto a fixed distance
    matrix.

    Parameters
    ----------
    metric: callable
        An object that implements a distance metric.
    sigma: float
        The log scale hyperparameter for the RBF Kernel.
    sigma_bounds: float
        The bounds for sigma.
    '''

    def __init__(self, D, sigma, sigma_bounds=(1e-3, 1e3),
                 sticky_cache=False):
        self.sigma = sigma
        self.sigma_bounds = sigma_bounds
        self.D = D

    def __call__(self, X, Y=None, eval_gradient=False):
        '''
        Parameters
        ----------
        eval_gradient: bool
            If true, also return the gradient of the weights with respect to
            the **log-scale** hyperparameters.
        '''
        d = self.D[X, :][:, X if Y is None else Y]
        w = np.exp(-0.5 * d**2 * self.sigma**-2)
        if Y is None:
            w[np.diag_indices_from(w)] = 0
        if eval_gradient:
            j = d**2 * w * self.sigma**-3
            return w, np.stack([j], axis=2)
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
        return np.log([self.sigma_bounds])
