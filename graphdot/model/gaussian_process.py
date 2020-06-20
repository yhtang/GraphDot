#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from scipy.optimize import minimize
from ._chol_solver import CholSolver


class GaussianProcessRegressor:
    """Gaussian process regression (GPR).

    Parameters
    ----------
    kernel: kernel instance
        The covariance function of the GP.
    alpha: float > 0, default = 1e-10
        Value added to the diagonal of the kernel matrix during fitting. Larger
        values correspond to increased noise level in the observations. A
        practical usage of this parameter is to prevent potential numerical
        stability issues during fitting, and ensures that the kernel matrix is
        always positive definite in the precense of duplicate entries and/or
        round-off error.
    optimizer: one of (str, True, None, callable)
        A string or callable that represents one of the optimizers usable in
        the scipy.optimize.minimize method.
        If None, no hyperparameter optimization will be carried out in fitting.
        If True, the optimizer will default to L-BFGS-B.
    normalize_y: boolean
        Whether to normalize the target values y so that the mean and variance
        become 0 and 1, respectively. Recommended for cases where zero-mean,
        unit-variance kernels are used. Normalisation will be reversed before
        the GP predictions are returned.
    """

    def __init__(self, kernel, alpha=1e-10, optimizer=None, normalize_y=False):
        self.kernel = kernel
        self.alpha = alpha
        self.optimizer = optimizer
        if optimizer is True:
            self.optimizer = 'L-BFGS-B'
        self.normalize_y = normalize_y

    def fit(self, X, y):
        """Train a GPR model with the given data.

        Parameters
        ----------
        X: list of objects or feature vectors.
            Input values of the training data.
        y: 1D array
            Output/target values of the training data.

        Returns
        -------
        self: returns an instance of self.
        """
        # set X
        self.X = X
        # set y
        if self.normalize_y is True:
            self.y_mean, self.y_std = np.mean(y), np.std(y)
            self.y = (y - self.y_mean) / self.y_std
        else:
            self.y_mean, self.y_std = 0, 1
            self.y = np.copy(y)
        # train model
        if self.optimizer:
            self._optimize_hyperparameters(X, y)
        self.K = K = self.kernel(X)
        K.flat[::len(K) + 1] += self.alpha
        self.L = CholSolver(K)
        self.Ky = self.L(y)
        return self

    def predict(self, Z, return_std=False, return_cov=False):
        """Predict using the trained GPR model.

        Parameters
        ----------
        Z: list of objects or feature vectors.
            Input values of the unknown data.
        return_std: boolean
            If True, the standard-deviations of the predictions at the query
            points are returned along with the mean.
        return_cov: boolean
            If True, the covariance of the predictions at the query points are
            returned along with the mean.

        Returns
        -------
        ymean: 1D array
            Mean of the predictive distribution at query points.
        std: 1D array
            Standard deviation of the predictive distribution at query points.
        cov: 2D matrix
            Covariance of the predictive distribution at query points.
        """
        if not hasattr(self, 'K'):
            raise RuntimeError('Model not trained.')
        Ks = self.kernel(Z, self.X)
        ymean = (Ks @ self.Ky) * self.y_std + self.y_mean
        if return_std is True:
            Kss = self.kernel.diag(Z)
            std = np.sqrt(np.maximum(Kss - (Ks @ self.L(Kss.T)).diagonal(), 0))
            return (ymean, std * self.y_std)
        elif return_cov is True:
            Kss = self.kernel(Z)
            cov = np.maximum(Kss - Ks @ self.L(Kss.T), 0)
            return (ymean, cov * self.y_std**2)
        else:
            return ymean

    def log_marginal_likelihood(self, theta=None, eval_gradient=False,
                                clone_kernel=True, verbose=False):
        """Returns log-marginal likelihood of a given set of log-scale
        hyperparameters for the training data as previously given by fit().

        Parameters
        ----------
        theta: array-like
            Kernel hyperparameters for which the log-marginal likelihood is
            to be evaluated. If None, the current hyperparameters will be used.
        eval_gradient: boolean
            If True, the gradient of the log-marginal likelihood with respect
            to the kernel hyperparameters at position theta will be returned
            alongside.
        clone_kernel: boolean
            If True, the kernel is copied so that probing with theta does not
            alter the trained kernel. If False, the kernel hyperparameters will
            be modified in-place.
        verbose: boolean
            If True, the log-likelihood value and its components will be
            printed to the screen.

        Returns
        -------
        log_likelihood: float
            Log-marginal likelihood of theta for training data.
        log_likelihood_gradient: 1D array
            Gradient of the log-marginal likelihood with respect to the kernel
            hyperparameters at position theta. Only returned when eval_gradient
            is True.
        """
        if theta is None:
            theta = self.kernel.theta

        if clone_kernel is True:
            kernel = self.kernel.clone_with_theta(theta)
        else:
            kernel = self.kernel
            kernel.theta = theta

        if eval_gradient is True:
            K, dK = kernel(self.X, eval_gradient=True)
        else:
            K = kernel(self.X)
        K.flat[::len(K) + 1] += self.alpha

        L = CholSolver(K)
        Ky = L(self.y)
        yKy = self.y @ Ky
        logdet = np.prod(np.linalg.slogdet(K))
        if verbose:
            print(
                'L %.5f = %.5f + %.5f' % (yKy + logdet, yKy, logdet),
                np.exp(theta)
            )

        if eval_gradient is True:
            D_theta = np.zeros_like(theta)
            for i, t in enumerate(theta):
                dk = dK[:, :, i]
                D_theta[i] = (L(dk).trace() - Ky @ dk @ Ky) * np.exp(t)
            return yKy + logdet, D_theta
        else:
            return yKy + logdet

    def _optimize_hyperparameters(self, X, y):
        opt = minimize(
            fun=lambda theta, self=self: self.log_marginal_likelihood(
                theta, eval_gradient=True, clone_kernel=False
            ),
            method=self.optimizer,
            x0=self.kernel.theta,
            bounds=self.kernel.bounds,
            jac=True,
            tol=1e-3,
        )

        if opt.success:
            self.kernel.theta = opt.x
        else:
            raise RuntimeError(
                f'Hyperparameter optimization did not converge, result:\n{opt}'
            )
