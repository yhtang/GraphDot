#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from scipy.optimize import minimize
from ._chol_solver import CholSolver


class GaussianProcessRegressor:
    def __init__(self, kernel, alpha=0, optimizer=None, normalize_y=False):
        self.kernel = kernel
        self.alpha = alpha
        self.optimizer = optimizer
        if optimizer is True:
            self.optimizer = 'L-BFGS-B'
        self.normalize_y = normalize_y

    def fit(self, X, y):
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
        if not hasattr(self, 'K'):
            raise RuntimeError('Model not trained.')
        Ks = self.kernel(Z, self.X)
        pred = (Ks @ self.Ky) * self.y_std + self.y_mean
        if return_std is True:
            Kss = self.kernel.diag(Z)
            std = np.sqrt(np.maximum(Kss - (Ks @ self.L(Kss.T)).diagonal(), 0))
            return (pred, std * self.y_std)
        elif return_cov is True:
            Kss = self.kernel(Z)
            cov = np.maximum(Kss - Ks @ self.L(Kss.T), 0)
            return (pred, cov * self.y_std**2)
        else:
            return pred

    def log_marginal_likelihood(self, theta, eval_gradient=False,
                                clone_kernel=True, verbose=False):
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
            D_theta = np.zeros(dK.shape[0])
            for i in range(dK.shape[0]):
                dk = dK[i, :, :]
                D_theta[i] = (L(dk).trace() - Ky @ dk @ Ky) * np.exp(theta[i])
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
