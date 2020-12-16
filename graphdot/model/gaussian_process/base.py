#!/usr/bin/env python
# -*- coding: utf-8 -*-
import itertools as it
import os
import pickle
import warnings
import numpy as np
from scipy.optimize import minimize
from graphdot.linalg.cholesky import CholSolver
from graphdot.linalg.spectral import pinvh
from graphdot.util.printer import markdown as mprint


class GaussianProcessRegressorBase:
    """Base class for all Gaussian process regression (GPR) models."""

    def __init__(self, kernel, normalize_y, regularization, kernel_options):
        self.kernel = kernel
        self.normalize_y = normalize_y
        self.regularization = regularization
        self.kernel_options = kernel_options

    @property
    def X(self):
        '''The input values of the training set.'''
        try:
            return self._X
        except AttributeError:
            raise AttributeError(
                'Training data does not exist. Please provide using fit().'
            )

    @X.setter
    def X(self, X):
        self._X = np.asarray(X)

    @property
    def y(self):
        '''The output/target values of the training set.'''
        try:
            return self._y * self._ystd + self._ymean
        except AttributeError:
            raise AttributeError(
                'Training data does not exist. Please provide using fit().'
            )

    @staticmethod
    def mask(iterable):
        mask = np.fromiter(
            map(lambda i: i is not None and np.isfinite(i), iterable),
            dtype=np.bool_
        )
        masked = np.fromiter(it.compress(iterable, mask), dtype=np.float)
        return mask, masked

    @y.setter
    def y(self, y):
        self._y_mask, y_masked = self.mask(y)
        if self.normalize_y is True:
            self._ymean, self._ystd = y_masked.mean(), y_masked.std()
            self._y = (y_masked - self._ymean) / self._ystd
        else:
            self._ymean, self._ystd = 0, 1
            self._y = y_masked

    def _regularize(self, K, alpha):
        if self.regularization in ['+', 'additive']:
            return K + alpha
        elif self.regularization in ['*', 'multiplicative']:
            return K * (1 + alpha)
        else:
            raise RuntimeError(
                f'Unknown regularization method {self.regularization}.'
            )

    def _gramian(self, alpha, X, Y=None, kernel=None, jac=False, diag=False):
        kernel = kernel or self.kernel
        if Y is None:
            if diag is True:
                return self._regularize(
                    kernel.diag(X, **self.kernel_options), alpha
                )
            else:
                if jac is True:
                    K, J = kernel(X, eval_gradient=True, **self.kernel_options)
                    K.flat[::len(K) + 1] = self._regularize(
                        K.flat[::len(K) + 1], alpha
                    )
                    return K, J
                else:
                    K = kernel(X, **self.kernel_options)
                    K.flat[::len(K) + 1] = self._regularize(
                        K.flat[::len(K) + 1], alpha
                    )
                    return K
        else:
            if diag is True:
                raise ValueError(
                    'Diagonal Gramian does not exist between two sets.'
                )
            else:
                if jac is True:
                    return kernel(X, Y, eval_gradient=True,
                                  **self.kernel_options)
                else:
                    return kernel(X, Y, **self.kernel_options)

    def _invert(self, K, rcond):
        try:
            return self._invert_cholesky(K)
        except np.linalg.LinAlgError:
            try:
                warnings.warn(
                    'Kernel matrix singular, falling back to pseudoinverse'
                )
                return self._invert_pseudoinverse(K, rcond)
            except np.linalg.LinAlgError:
                raise np.linalg.LinAlgError(
                    'The kernel matrix is likely corrupted with NaNs and Infs '
                    'because a pseudoinverse could not be computed.'
                )

    def _invert_cholesky(self, K):
        return CholSolver(K), np.prod(np.linalg.slogdet(K))

    def _invert_pseudoinverse(self, K, rcond):
        return pinvh(K, rcond=rcond, mode='clamp', return_nlogdet=True)

    def _hyper_opt(self, method, fun, xgen, tol, verbose):
        opt = None

        for x in xgen:
            if verbose:
                mprint.table_start()

            opt_local = minimize(
                fun=fun,
                method=method,
                x0=x,
                bounds=self.kernel.bounds,
                jac=True,
                tol=tol,
            )

            if not opt or (opt_local.success and opt_local.fun < opt.fun):
                opt = opt_local

        return opt

    def save(self, path, filename='model.pkl', overwrite=False):
        """Save the trained GaussianProcessRegressor with the associated data
        as a pickle.

        Parameters
        ----------
        path: str
            The directory to store the saved model.
        filename: str
            The file name for the saved model.
        overwrite: bool
            If True, a pre-existing file will be overwritten. Otherwise, a
            runtime error will be raised.
        """
        f_model = os.path.join(path, filename)
        if os.path.isfile(f_model) and not overwrite:
            raise RuntimeError(
                f'Path {f_model} already exists. To overwrite, set '
                '`overwrite=True`.'
            )
        store = self.__dict__.copy()
        store['theta'] = self.kernel.theta
        store.pop('kernel', None)
        pickle.dump(store, open(f_model, 'wb'), protocol=4)

    def load(self, path, filename='model.pkl'):
        """Load a stored GaussianProcessRegressor model from a pickle file.

        Parameters
        ----------
        path: str
            The directory where the model is saved.
        filename: str
            The file name for the saved model.
        """
        f_model = os.path.join(path, filename)
        store = pickle.load(open(f_model, 'rb'))
        theta = store.pop('theta')
        self.__dict__.update(**store)
        self.kernel.theta = theta
