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

    def fit(self, X, y, loss='loocv2', tol=1e-5, repeat=1, theta_jitter=1.0,
            verbose=False):
        '''Train the Gaussian field model.

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

            - 'ale' or 'average-label-entropy': average label entropy. Only
            works if the labels are 0/1 binary.
            - 'loocv1' or 'loocv2': the leave-one-out cross validation of the
            labeled samples as measured in L1/L2 norm.

        tol: float
            Tolerance for termination.
        repeat: int
            Repeat the hyperparameter optimization by the specified number of
            times and return the best result.
        theta_jitter: float
            Standard deviation of the random noise added to the initial
            logscale hyperparameters across repeated optimization runs.

        Returns
        -------
        self: GaussianFieldRegressor
            returns an instance of self.
        '''
        assert len(X) == len(y)
        X = np.asarray(X)
        y = np.asarray(y, dtype=np.float)

        if hasattr(self.weight, 'theta') and self.optimizer:
            try:
                objective = {
                    'ale': self.average_label_entropy,
                    'average-label-entropy': self.average_label_entropy,
                    'loocv1': self.loocv_error_1,
                    'loocv2': self.loocv_error_2,
                }[loss]
            except KeyError:
                raise RuntimeError(f'Unknown loss function \'{loss}\'')

            def xgen(n):
                x0 = self.weight.theta.copy()
                yield x0
                yield from x0 + theta_jitter * np.random.randn(n - 1, len(x0))

            opt = self._hyper_opt(
                method=self.optimizer,
                fun=lambda theta, objective=objective: objective(
                    X, y, theta=theta, eval_gradient=True, verbose=verbose
                ),
                xgen=xgen(repeat), tol=tol, verbose=verbose
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

        return self

    def predict(self, X, y, return_influence=False):
        '''Make predictions for the unlabeled elements in y.

        Parameters
        ----------
        X: 2D array or list of objects
            Feature vectors or other generic representations of input data.
        y: 1D array
            Label of each data point. Values of None or NaN indicates
            missing labels that will be filled in by the model.
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
        '''
        assert len(X) == len(y)
        X = np.asarray(X)
        y = np.asarray(y, dtype=np.float)

        z = y.copy()
        if return_influence is True:
            z[~np.isfinite(y)], influence = self._predict(
                X, y, return_influence=True
            )
            return z, influence
        else:
            z[~np.isfinite(y)] = self._predict(X, y, return_influence=False)
            return z

    def fit_predict(self, X, y, loss='average-label-entropy', tol=1e-5,
                    repeat=1, theta_jitter=1.0, return_influence=False,
                    verbose=False):
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

            - 'ale' or 'average-label-entropy': average label entropy. Only
            works if the labels are 0/1 binary.
            - 'loocv1' or 'loocv2': the leave-one-out cross validation of the
            labeled samples as measured in L1/L2 norm.

        tol: float
            Tolerance for termination.
        repeat: int
            Repeat the hyperparameter optimization by the specified number of
            times and return the best result.
        theta_jitter: float
            Standard deviation of the random noise added to the initial
            logscale hyperparameters across repeated optimization runs.
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
        '''
        self.fit(
            X, y, loss=loss, tol=tol, repeat=repeat,
            theta_jitter=theta_jitter, verbose=verbose
        )

        return self.predict(X, y, return_influence=return_influence)

    def _hyper_opt(self, method, fun, xgen, tol, verbose):
        opt = None

        for x in xgen:
            if verbose:
                mprint.table_start()

            opt_local = minimize(
                fun=fun,
                method=method,
                x0=x,
                bounds=self.weight.bounds,
                jac=True,
                tol=tol
            )

            if not opt or (opt_local.success and opt_local.fun < opt.fun):
                opt = opt_local

        return opt

    def _predict(self, X, y, return_influence=False):
        labeled = np.isfinite(y)
        f_l = y[labeled]
        if len(f_l) == len(y):
            raise RuntimeError(
                'All samples are labeled, no predictions will be made.'
            )
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
        if len(f_l) == len(y):
            raise RuntimeError(
                'All samples are labeled, no predictions will be made.'
            )
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
        verbose: bool
            If true, print out some additional information as a markdown table.

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

    def loocv_error(self, X, y, p=2, theta=None, eval_gradient=False,
                    verbose=False):
        '''Evaluate the leave-one-out cross validation error and gradient.

        Parameters
        ----------
        X: 2D array or list of objects
            Feature vectors or other generic representations of input data.
        y: 1D array
            Label of each data point. Values of None or NaN indicates
            missing labels that will be filled in by the model.
        p: float > 1
            The order of the p-norm for LOOCV error.
        theta: 1D array
            Hyperparameters for the weight class.
        eval_gradients:
            Whether or not to evaluate the gradient of the average label
            entropy with respect to weight hyperparameters.
        verbose: bool
            If true, print out some additional information as a markdown table.

        Returns
        -------
        err: 1D array
            LOOCV Error
        grad: 1D array
            Gradient with respect to the hyperparameters.
        '''
        if theta is not None:
            self.weight.theta = theta

        labeled = np.isfinite(y)
        y = y[labeled]
        n = len(y)
        t_metric = time.perf_counter()
        if eval_gradient is True:
            W, dW = self.weight(X[labeled], eval_gradient=True)
        else:
            if self.weight == 'precomputed':
                W = X[labeled, :][:, labeled]
            else:
                W = self.weight(X[labeled])
        t_metric = time.perf_counter() - t_metric

        t_chain = time.perf_counter()
        W += self.smoothing
        D = W.sum(axis=1)
        P = (1 / D)[:, None] * W
        e = y - P @ y
        loocv_error = np.mean(np.abs(e)**p)**(1/p)
        if eval_gradient is True:
            derr_de = (
                np.mean(np.abs(e)**p)**(1/p - 1)
                * np.abs(e)**(p - 1) * np.sign(e) / n
            )
            de_dW = (
                -np.einsum('ip,q,i->ipq', np.eye(n), y, 1 / D)
                + np.diag(1 / D**2 * (W @ y))[:, :, None]
            )
            derr_dW = np.einsum('i, ipq->pq', derr_de, de_dW)
            derr_dtheta = np.einsum('pq,pqr->r', derr_dW, dW)
            retval = (loocv_error, derr_dtheta)
        else:
            retval = loocv_error
        t_chain = time.perf_counter() - t_chain

        if verbose and eval_gradient is True:
            mprint.table(
                ('LOOCV Err.', '%12.5g', loocv_error),
                ('Gradient', '%12.5g', np.linalg.norm(derr_dtheta)),
                ('Metric time', '%12.2g', t_metric),
                ('BackProp time', '%14.2g', t_chain),
            )

        return retval

    def loocv_error_1(self, X, y, **kwargs):
        '''Leave-one-out cross validation error measured in L1 norm.
        Equivalent to :py:method:`loocv_error(X, y, p=1, **kwargs)`.
        '''
        return self.loocv_error(X, y, p=1, **kwargs)

    def loocv_error_2(self, X, y, **kwargs):
        '''Leave-one-out cross validation error measured in L2 norm.
        Equivalent to :py:method:`loocv_error(X, y, p=2, **kwargs)`.
        '''
        return self.loocv_error(X, y, p=2, **kwargs)
