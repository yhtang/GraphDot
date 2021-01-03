#!/usr/bin/env python
# -*- coding: utf-8 -*-
import time
import numpy as np
from scipy.optimize import minimize
from graphdot.linalg.cholesky import CholSolver
from graphdot.util.printer import markdown as mprint


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
        assert smoothing >= 0, "Smoothing must be no less than 0."
        self.weight = weight
        self.optimizer = optimizer
        if optimizer is True:
            self.optimizer = 'L-BFGS-B'
        self.smoothing = smoothing

    def fit_predict(self, X, y, loss='average-label-entropy', options=None,
                    return_influence=False, verbose=False):
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

        '''The 'fit' part'''
        if hasattr(self.weight, 'theta') and self.optimizer:
            try:
                objective = {
                    'ale': self.average_label_entropy,
                    'average-label-entropy': self.average_label_entropy,
                    'laplacian': self.laplacian,
                }[loss]
            except KeyError:
                raise RuntimeError(f'Unknown loss function \'{loss}\'')

            if verbose is True:
                mprint.table_start()

            opt = minimize(
                fun=lambda theta, objective=objective: objective(
                    X, y, theta, eval_gradient=True, verbose=verbose
                ),
                method=self.optimizer,
                x0=self.weight.theta,
                bounds=self.weight.bounds,
                jac=True,
                tol=1e-5,
                options=options
            )
            if verbose:
                print(f'Optimization result:\n{opt}')
            if opt.success:
                self.weight.theta = opt.x
            else:
                raise RuntimeError(
                    f'Optimizer did not converge, got:\n'
                    f'{opt}'
                )

        '''The 'predict' part'''
        z = y.copy()
        if return_influence is True:
            z[~np.isfinite(y)], influence = self._predict(
                X, y, return_influence=True
            )
            return z, influence
        else:
            z[~np.isfinite(y)] = self._predict(X, y, return_influence=False)
            return z

    def _predict(self, X, y, return_influence=False):
        labeled = np.isfinite(y)
        f_l = y[labeled]
        if self.weight == 'precomputed':
            W_uu = X[~labeled, :][:, ~labeled] + self.smoothing
            W_ul = X[~labeled, :][:, labeled] + self.smoothing
        else:
            W_uu = self.weight(X[~labeled]) + self.smoothing
            W_ul = self.weight(X[~labeled], X[labeled]) + self.smoothing
        D = W_uu.sum(axis=1) + W_ul.sum(axis=1)

        try:
            L_inv = CholSolver(np.diag(D) - W_uu)
        except np.linalg.LinAlgError:
            raise RuntimeError(
                'The Graph Laplacian is not positive definite. Some'
                'weights on edges may be invalid.'
            )

        if return_influence is True:
            influence = L_inv @ W_ul
            f_u = influence @ f_l
            return f_u, influence
        else:
            f_u = L_inv @ (W_ul @ f_l)
            return f_u

    def _predict_gradient(self, X, y):
        t_metric = time.perf_counter()
        labeled = np.isfinite(y)
        f_l = y[labeled]
        W_uu, dW_uu = self.weight(X[~labeled], eval_gradient=True)
        W_ul, dW_ul = self.weight(X[~labeled], X[labeled], eval_gradient=True)
        W_uu += self.smoothing
        W_ul += self.smoothing
        D = W_uu.sum(axis=1) + W_ul.sum(axis=1)
        t_metric = time.perf_counter() - t_metric

        t_solve = time.perf_counter()
        try:
            L_inv = CholSolver(np.diag(D) - W_uu).todense()
        except np.linalg.LinAlgError:
            raise RuntimeError(
                'The Graph Laplacian is not positive definite. Some'
                'weights on edges may be invalid.'
            )
        t_solve = time.perf_counter() - t_solve

        t_chain = time.perf_counter()
        f_u = L_inv @ (W_ul @ f_l)
        dL_inv = L_inv * f_u
        df_u = (
            np.einsum('im,n,mnj->ij', L_inv, f_u, dW_uu, optimize=True)
            + np.einsum('im,n,mnj->ij', L_inv, f_l, dW_ul, optimize=True)
            - np.einsum('imn,mnj->ij', dL_inv[:, :, None], dW_uu)
            - np.einsum('imn,mnj->ij', dL_inv[:, :, None], dW_ul)
        )
        t_chain = time.perf_counter() - t_chain

        return f_u, df_u, t_metric, t_solve, t_chain

    def average_label_entropy(self, X, y, theta=None, eval_gradient=False,
                              verbose=False):
        '''Evaluate the average label entropy of the Gaussian field model on a
        dataset.

        Parameters
        ----------
        X: 2D array or list of objects
            Feature vectors or other generic representations of input data.
        y: 1D array
            Label of each data point. Values of None or NaN indicates
            missing labels that will be filled in by the model.
        theta: 1D array
            Hyperparameters for the weight class.
        eval_gradients:
            Whether or not to evaluate the gradient of the average label
            entropy with respect to weight hyperparameters.

        Returns
        -------
        average_label_entropy: float
            The average label entropy of the Gaussian field prediction on the
            unlabeled nodes.
        grad: 1D array
            Gradient with respect to the hyperparameters.
        '''
        if theta is not None:
            self.weight.theta = theta

        if eval_gradient is True:
            z, dz, t_metric, t_solve, t_chain = self._predict_gradient(
                X, y
            )
        else:
            z = self._predict(X, y)
        loss = -np.mean(z * np.log(z) + (1 - z) * np.log(1 - z))
        if eval_gradient is True:
            dloss = np.log(z) - np.log(1 - z)
            grad = -np.mean(dloss * dz.T, axis=1) * np.exp(self.weight.theta)
            retval = (loss, grad)
        else:
            retval = loss

        if verbose and eval_gradient is True:
            mprint.table(
                ('Avg.Entropy', '%12.5g', loss),
                ('Gradient', '%12.5g', np.linalg.norm(grad)),
                ('Metric time', '%12.2g', t_metric),
                ('Solver time', '%12.2g', t_solve),
                ('BackProp time', '%14.2g', t_chain),
            )

        return retval

    def laplacian(self, X, y, theta=None):
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
        if theta is not None:
            self.weight.theta = theta

        labeled = np.isfinite(y)
        y = y[labeled]
        if self.weight == 'precomputed':
            W = X[labeled, :][:, labeled]
        else:
            W = self.weight(X[labeled])
        W += self.smoothing
        D = W.sum(axis=1)
        h = D * y - W @ y
        h_norm = np.linalg.norm(h, ord=2)
        return h_norm
