#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from ._chol_solver import CholSolver


class GaussianProcessRegressor:
    def __init__(self, kernel, alpha=0, optimizer=None, normalize_y=False):
        self.kernel = kernel
        self.alpha = alpha
        self.optimizer = optimizer
        self.normalize_y = normalize_y

    def fit(self, X, y):
        self.X = X
        self.K = K = self.kernel(X)
        K.flat[::len(K) + 1] += self.alpha
        self.L = CholSolver(K)
        if self.normalize_y is True:
            m, s = np.mean(y), np.std(y)
            self.y = (y - m) / s
            self.y_scaler = lambda y0, m=m, s=s: y0 * s + m
        else:
            self.y = np.copy(y)
            self.y_scaler = lambda y0: y0
        self.Ky = self.L(y)
        return self

    def predict(self, Z, return_std=False, return_cov=False):
        if not hasattr(self, 'K'):
            raise RuntimeError('Model not trained.')
        Ks = self.kernel(Z, self.X)
        pred = self.y_scaler(Ks @ self.Ky)
        if return_std is True:
            Kss = self.kernel.diag(Z)
            std = np.sqrt(np.maximum(Kss - (Ks @ self.L(Kss.T)).diagonal(), 0))
            return (pred, std)
        elif return_cov is True:
            Kss = self.kernel(Z)
            cov = np.maximum(Kss - Ks @ self.L(Kss.T), 0)
            return (pred, cov)
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
            print('L %.5f = %.5f + %.5f' % (yKy + logdet, yKy, logdet), theta)

        if eval_gradient is True:
            D_theta = []
            for i in range(dK.shape[0]):
                dk = dK[i, :, :]
                dt = (L(dk).trace() - Ky @ dk @ Ky) * np.exp(theta[i])
                D_theta.append(dt)
            return yKy + logdet, np.array(D_theta)
        else:
            return yKy + logdet
