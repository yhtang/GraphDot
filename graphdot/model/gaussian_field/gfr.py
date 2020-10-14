#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from scipy.sparse.linalg import bicgstab
from scipy.optimize import minimize


class GaussianFieldRegressor:
    '''Semi-supervised learning and prediction of missing labels of continuous
    value on a graph. Reference: Zhu, Ghahramani, Lafferty. ICML 2003

    Parameters
    ----------
    weight: functor or 'precomputed'
        A function that implements a weight function that converts distance
        to graph edge weights. The value of a weight function should generally
        decay with distance. If weight is 'precomputed', then the result as
        computed by `metric` will be directly used as weight.
    optimizer: one of (sstr, True, None, callable)
        A string or callable that represents one of the optimizers usable in
        the scipy.optimize.minimize method.
        if None, no hyperparameter optimization will be carried out in fitting.
        If True, the optimizer will default to L-BFGS-B.
    smoothing: float in [0, 1)
        Float that controls the smoothing of the transition matrix. Used for
        regularization.
    eps: float in [0, 1)
        Controls the step size for approximating gradients with finite
        difference.
    eta: float in [0, 1)
        Controls a dongle implementation. When eta > 0, each node will have
        a copy made attached with an edge weight of eta * the total weight
        of all other nodes attached to the original. The model will then
        treat the original node as an unlabeled node. Used for regularization.
    '''

    def __init__(self, weight, optimizer=None, smoothing=1e-3, eps=1e-3, eta=0):
        self.weight = weight
        self.optimizer = optimizer
        if optimizer is True:
            self.optimizer = 'L-BFGS-B'
        self.data = None
        self.labels = None
        self.eps = eps
        if smoothing < 0 or smoothing > 1:
            raise RuntimeError("Smoothing must be between 0 and 1 inclusive.")
        self.smoothing = smoothing
        if eta < 0 or eta > 1:
            raise RuntimeError("Eta must be between 0 and 1 inclusive.")
        self.eta = eta

    def set_model(self, X, labels):
        '''Input training data into the model

        Parameters
        ----------
        X: 2D array or list of objects
            Feature vectors or other generic representaations of labeled data.
        labels: 1D array
            Labels/target values.
        '''
        self.X = X
        self.labels = labels

    def fit(self, Z=None, y=None, loss=None, options=None):
        '''Train the Gaussian field model. If the loss is unspecified,
        average label entropy will be used by default.

        Parameters
        ----------
        X: 2D array or list of objects
            Feature vectors or other generic representations of labeled data.
        y: 1D array
            Labels/target values.
        append_data: boolean
            Whether or not to add fitted data to the model
        '''
        if len(self.X) == 0:
            raise RuntimeError("Missing Training Data")
        if len(self.labels) == 0:
            raise RuntimeError("Missing Training Labels")
        if self.weight.theta is not None and self.optimizer:
            if loss == 'mse':
                objective = self.squared_error
            elif loss == 'cross_entropy':
                objective = self.cross_entropy
            elif loss == 'laplacian':
                objective = self.laplacian_error
            else:
                objective = self.average_label_entropy
            opt = minimize(
                fun=lambda theta, objective=objective: objective(
                    Z, y, theta, eval_gradient=True
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

    def predict(self, Z, display=False):
        '''Predict using the trained Gaussian field model.

        Parameters
        ----------
        Z: 2D array or list of objects
            Feature vectors or other generic representations of unlabeled data.

        display: boolean
            Whether or not to display the label weightings and predictive
            uncertainty.

        Returns
        -------
        y: 1D array
            Predicted values of the input data.
        influence_matrix: 2D array
            Contributions from each labeled sample for each predicted label.
        predictive_uncertainty: 1D array
            Weighted Standard Deviation of the predicted labels.
        '''
        if len(self.X) == 0:
            raise RuntimeError("Missing Training Data")
        if len(self.labels) == 0:
            raise RuntimeError("Missing Training Labels")
        if len(Z) == 0:
            return []

        X = self.X
        labels = self.labels
        weight = self.weight
        smoothing = self.smoothing
        L = len(X)
        u = len(Z)
        n = L + u

        if self.eta:
            if isinstance(Z, pd.DataFrame):
                data = pd.concat([X, Z], ignore_index=True)
            elif isinstance(Z, np.ndarray):
                data = np.concatenate([X, Z])
            W_uu = weight(data, data)
            d = np.sum(W_uu[:L], axis=1) - W_uu.diagonal()[:L]
            W_ul = np.zeros((n, L))
            np.fill_diagonal(W_ul, self.eta * d/(1 - self.eta))
            D_uu_inv = np.diag(1/(np.sum(W_ul, axis=1) + np.sum(W_uu, axis=1)))
            P_uu = smoothing / (n + L) + (1 - smoothing) * D_uu_inv @ W_uu
            P_ul = smoothing / (n + L) + (1 - smoothing) * D_uu_inv @ W_ul

        else:
            W_ul = weight(Z, X)
            W_uu = weight(Z)
            D_uu_inv = np.diag(1/(np.sum(W_ul, axis=1) + np.sum(W_uu, axis=1)))
            P_uu = smoothing / n + (1 - smoothing) * D_uu_inv @ W_uu
            P_ul = smoothing / n + (1 - smoothing) * D_uu_inv @ W_ul

        if self.eta:
            A = np.eye(n) - P_uu
        else:
            A = np.eye(u) - P_uu
        B = P_ul
        f_u, _ = bicgstab(A, B @ labels)
        result = f_u
        if display:
            weight_matrix = np.linalg.solve(A, B)
            influence_matrix = weight_matrix * (2 * labels - 1)
            raw_mean = weight_matrix * (2 * (labels - f_u[:, None]))**2
            predictive_uncertainty = np.sum(raw_mean, axis=1)**0.5
            result = f_u, influence_matrix, predictive_uncertainty
        return result

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
