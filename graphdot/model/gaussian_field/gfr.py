#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from scipy.sparse.linalg import bicgstab, LinearOperator
from scipy.optimize import minimize


class GaussianFieldRegressor:
    '''Semi-supervised learning and prediction of missing labels of continuous
    value on a graph. Reference: Zhu, Ghahramani, Lafferty. ICML 2003

    Parameters
    ----------
    weight: callable or 'precomputed'
        A function that implements a weight function that converts distance
        matrices to weight matrices. The value of a weight function should
        generally decay with distance. If weight is 'precomputed', then the
        result returned by `metric` will be directly used as weight.
    optimizer: one of (str, True, None, callable)
        A string or callable that represents one of the optimizers usable in
        the scipy.optimize.minimize method.
        if None, no hyperparameter optimization will be carried out in fitting.
        If True, the optimizer will default to L-BFGS-B.
    smoothing: float in [0, 1)
        Controls the strength of regularization via the smoothing of the
        transition matrix.
    '''

    def __init__(self, weight, optimizer=None, smoothing=1e-3):
        assert smoothing >= 0 and smoothing < 1, "Smoothing must be in [0, 1)."
        self.weight = weight
        self.optimizer = optimizer
        if optimizer is True:
            self.optimizer = 'L-BFGS-B'
        self.smoothing = smoothing

    def fit_predict(self, X, y, loss='average_label_entropy', options=None,
                    return_influence=False):
        '''Train the Gaussian field model and make predictions for the
        unlabeled nodes.

        Parameters
        ----------
        X: 2D array or list of objects
            Feature vectors or other generic representations of input data.
        y: 1D array
            Label of each data point. Values of None or NaN indicates
            missing labels that will be filled in by the model.
        loss: str
            The loss function to be used to optimizing the hyperparameters.
            Options are:

            - 'ale' or 'average-label-entropy': average label entropy. Suitable
            for binary 0/1 labels.
            - 'laplacian': measures how well the known labels conform to the
            graph Laplacian operator. Suitable for continuous labels.

        return_influence: bool
            If True, also returns the contributions of each labeled sample to
            each predicted label as an 'influence matrix'.

        Returns
        -------
        z: 1D array
            Node labels with missing ones filled in by prediction.
        influence_matrix: 2D array
            Contributions of each labeled sample to each predicted label. Only
            returned if ``return_influence`` is True.
        # predictive_uncertainty: 1D array
        #     Weighted Standard Deviation of the predicted labels.
        '''
        assert len(X) == len(y)
        X = np.asarray(X)
        y = np.asarray(y, dtype=np.float)
        labeled = np.isfinite(y)

        if hasattr(self.weight, 'theta') and self.optimizer:
            try:
                objective = {
                    'mse': self.squared_error,
                    'cross_entropy': self.cross_entropy,
                    'laplacian': self.laplacian_error,
                    'average_label_entropy': self.average_label_entropy,
                }[loss]
            except KeyError:
                raise RuntimeError(f'Unknown loss function \'{loss}\'')
            # TODO: include smoothing and dongle as hyperparameters?
            opt = minimize(
                fun=lambda theta, objective=objective: objective(
                    self.X, self.y, self.labeled, theta, eval_gradient=True
                ),
                method=self.optimizer,
                x0=self.weight.theta,
                bounds=self.weight.bounds,
                jac=True,
                tol=1e-5,
                options=options
            )
            if opt.success:
                self.weight.theta = opt.x
            else:
                raise RuntimeError(
                    f'Optimizer did not converge, got:\n'
                    f'{opt}'
                )

        n = len(X)
        W = self.weight(X[~labeled], X)
        D_inv = 1 / W.sum(axis=1)
        P = self.smoothing / n + (1 - self.smoothing) * (D_inv[:, None] * W)
        P_uu = P[:, ~labeled]
        prediction, _ = bicgstab(
            LinearOperator(P_uu.shape, lambda v: v - P_uu @ v),
            P[:, labeled] @ y[labeled]
        )

        z = y.copy()
        z[~labeled] = prediction

        # if display:
        #     weight_matrix = np.linalg.solve(A, B)
        #     influence_matrix = weight_matrix * (2 * labels - 1)
        #     raw_mean = weight_matrix * (2 * (labels - f_u[:, None]))**2
        #     predictive_uncertainty = np.sum(raw_mean, axis=1)**0.5
        #     result = f_u, influence_matrix, predictive_uncertainty

        return z

    def squared_error(self, Z, y, theta=None,
                      eval_gradient=False):
        '''Evaluate the Mean Sqared Error and gradient using the trained Gaussian
        field model on a dataset.

        Parameters
        ----------
        Z: 2D array or list of objects
            Feature vectors or other generic representations of unlabeled data.

        y: 1D array or list of objects
            Label values for Data.

        theta: 1D array or list of objects
            Hyperparameters for the weight class

        eval_gradients:
            Whether or not to evaluate the gradients.

        Returns
        -------
        err: 1D array
            Mean Squared Error

        grad: 1D array
            Gradient with respect to the hyperparameters.
        '''

        if len(self.X) == 0:
            raise RuntimeError("Missing Training Data")
        if len(self.labels) == 0:
            raise RuntimeError("Missing Training Labels")
        if theta is not None:
            self.weight.theta = theta

        predictions = self.predict(Z)

        f_u = predictions[-len(Z):]
        e = f_u - y
        err = 0.5 * np.sum(e**2)/len(f_u)
        if eval_gradient is True:
            grad = np.zeros_like(self.weight.theta)
            for i in range(len(self.weight.theta)):
                eps = self.eps
                self.weight.theta[i] += eps
                f1 = self.squared_error(Z, y, theta)
                self.weight.theta -= 2 * eps
                f2 = self.squared_error(Z, y, theta)
                self.weight.theta[i] += eps
                grad[i] = (f1 - f2)/(2 * eps)
            return err, grad
        else:
            return err

    def cross_entropy(self, Z, y, theta=None, eval_gradient=False):
        '''Evaluate the Cross Entropy gradient using the trained Gaussian field
        model on a dataset.

        Parameters
        ----------
        X: 2D array or list of objects
            Feature vectors or other generic representations of unlabeled data.

        y: 1D array or list of objects
            Label values for Data.

        theta: 1D array or list of objects
            Hyperparameters for the weight class

        eval_gradients:
            Whether or not to evaluate the gradients.

        Returns
        -------
        y: 1D array
            Predicted values of the input data.

        grad: 1D array
            Gradient with respect to the hyperparameters.
        '''
        if len(self.X) == 0:
            raise RuntimeError("Missing Training Data")
        if len(self.labels) == 0:
            raise RuntimeError("Missing Training Labels")

        if theta is not None:
            self.weight.theta = theta

        f_u = self.predict(Z)
        err = -y @ np.log(f_u) - (1 - y) @ np.log(1 - f_u)
        if eval_gradient is True:
            grad = np.zeros_like(self.weight.theta)
            for i in range(len(self.weight.theta)):
                eps = self.eps
                self.weight.theta[i] += eps
                f1 = self.cross_entropy(Z, y, theta)
                self.weight.theta -= 2 * eps
                f2 = self.cross_entropy(Z, y, theta)
                self.weight.theta[i] += eps
                grad[i] = (f1 - f2)/(2 * eps)
            return err, grad
        else:
            return err

    def average_label_entropy(self, Z, y, theta=None,
                              eval_gradient=False):
        '''Evaluate the Average Label Entropy gradient using the trained Gaussian
        field model on a dataset.

        Parameters
        ----------
        Z: 2D array or list of objects
            Feature vectors or other generic representations of unlabeled data.

        y: 1D array or list of objects
            Label values for Data.

        theta: 1D array or list of objects
            Hyperparameters for the weight class

        eval_gradients:
            Whether or not to evaluate the gradients.

        Returns
        -------
        err: 1D array
            Average Label Entropy

        grad: 1D array
            Gradient with respect to the hyperparameters.
        '''
        if len(self.X) == 0:
            raise RuntimeError("Missing Training Data")
        if len(self.labels) == 0:
            raise RuntimeError("Missing Training Labels")

        if theta is not None:
            self.weight.theta = theta

        predictions = self.predict(Z)
        f_u = predictions[-len(Z):]
        err = (-f_u @ np.log(f_u) - (1 - f_u) @ np.log(1 - f_u))/len(f_u)
        if eval_gradient is True:
            grad = np.zeros_like(self.weight.theta)
            for i in range(len(self.weight.theta)):
                eps = self.eps
                self.weight.theta[i] += eps
                f1 = self.average_label_entropy(Z, y, theta)
                self.weight.theta -= 2 * eps
                f2 = self.average_label_entropy(Z, y, theta)
                self.weight.theta[i] += eps
                grad[i] = (f1 - f2)/(2 * eps)
            return err, grad
        else:
            return err

    def laplacian_error(self, Z=None, y=None, theta=None, eval_gradient=False):
        '''Evaluate the Laplacian Error and gradient using the trained Gaussian
        field model on a dataset.

        Parameters
        ----------
        theta: 1D array or list of objects
            Hyperparameters for the weight class

        eval_gradients:
            Whether or not to evaluate the gradients.

        Returns
        -------
        err: 1D array
            Laplacian Error

        grad: 1D array
            Gradient with respect to the hyperparameters.
        '''
        if len(self.X) == 0:
            raise RuntimeError("Missing Training Data")
        if len(self.labels) == 0:
            raise RuntimeError("Missing Training Labels")
        if theta is not None:
            self.weight.theta = theta
        smoothing = self.smoothing
        weight = self.weight
        labels = self.labels
        X = self.X
        W = weight(X)
        U_ll = np.full((len(labels), len(labels)), 1/len(labels))
        D_inv = np.diag(1/np.sum(W, axis=1))
        h = (smoothing * (U_ll @ labels)) + \
            ((1 - smoothing) * D_inv @ (W @ labels))
        err = h @ h
        if eval_gradient is True:
            grad = np.zeros_like(self.weight.theta)
            for i in range(len(self.weight.theta)):
                eps = self.eps
                self.weight.theta[i] += eps
                f1 = self.laplacian_error(theta)
                self.weight.theta -= 2 * eps
                f2 = self.laplacian_error(theta)
                self.weight.theta[i] += eps
                grad[i] = (f1 - f2)/(2 * eps)
            return err, grad
        else:
            return err
