#!/usr/bin/env python
# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
import numpy as np
from graphdot.util.cookie import VolatileCookie


class Weight(ABC):

    @abstractmethod
    def __call__(self, X, Y=None, eval_gradient=False):
        '''Computes the weight matrix and optionally its gradient with respect
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

        Returns
        -------
        weight_matrix: 2D ndarray
            A weight matrix between the datasets.
        weight_matrix_gradients: 3D ndarray
            A tensor where the i-th frontal slide [:, :, i] contain the partial
            derivative of the weight matrix with respect to the i-th
            hyperparameter.
        '''
        pass

    @property
    @abstractmethod
    def theta(self):
        '''An ndarray of all the hyperparameters in log scale.'''
        pass

    @theta.setter
    @abstractmethod
    def theta(self, values):
        '''Set the hyperparameters from an array of log-scale values.'''
        pass

    @property
    @abstractmethod
    def bounds(self):
        '''The log-scale bounds of the hyperparameters as a 2D array.'''
        pass


class RBFOverHausdorff(Weight):
    '''Compute weights by applying an RBF onto the Hausdorff distance as
    derived from the graph kernel. Trained using an log scale sigma param.

    Parameters
    ----------
    metric: object
        An object that implements a distance metric between graph
        objects. This object may have hyperparameters to optimize for and
        return gradients with respect to.
    sigma: 1D array
        The log scale hyperparameter for the RBF Kernel.
    s_bounds: 2D array
        The bounds for sigma.
    graphs: container of graphs
        A container of graphs.
    metric: functor
        A functor that takes in graphs and outputs a distance matrix.
    compute_D_from_graphs: bool
        Whether or not to compute the D matrix during initialization.
    '''

    def __init__(self, sigma, sigma_bounds, graphs, metric,
                 compute_D_from_graphs=True):
        if not isinstance(sigma, np.ndarray):
            RuntimeError("Sigma is not an ndarray.")
        self.sigma = sigma
        self.sigma_bounds = sigma_bounds
        self.cookie = VolatileCookie()
        self.D = None
        if compute_D_from_graphs:
            self.set_D(graphs, metric)

    def set_D(self, graphs, metric):
        '''Compute D matrix from given graphs.

        Parameters
        ----------
        graphs: container
            A container of graphs.
        metric: functor
            A functor that takes in graphs and outputs a distance matrix
        '''
        for index, graph in enumerate(graphs):
            self.cookie[graph] = index
        self.D = metric(graphs)

    def __call__(self, X, Y=None, eval_gradient=False):
        '''A concrete implementation of the abstract method from the base
        class.

        Parameters
        ----------
        X, Y, eval_gradient: as previously defined.
            As defined in the base class.
        '''
        if self.D is None:
            raise RuntimeError("D matrix not computed. Call set_D")
        X_indices = [self.cookie[_] for _ in X]
        if Y is not None:
            Y_indices = [self.cookie[_] for _ in Y]
        else:
            Y_indices = [self.cookie[_] for _ in X]
        m = self.D[X_indices][:, Y_indices]
        s = np.exp(self.theta[-1])
        w = np.exp(-(m/s)**2)
        if eval_gradient:
            return w, np.array([m**2/self.sigma[0]**3]) * w
        else:
            return w

    @property
    def theta(self):
        '''The hyperparameter for the RBF kernel.'''
        return self.sigma

    @theta.setter
    def theta(self, values):
        self.sigma = values

    @property
    def bounds(self):
        '''The bounds for theta.'''
        return self.sigma_bounds

    @bounds.setter
    def bounds(self, values):
        self.sigma_bounds = values
