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
        unit-variance kernels are used. The normalization will be
        reversed when the GP predictions are returned.
    kernel_options: dict, optional
        A dictionary of additional options to be passed along when applying the
        kernel to data.
    """

    def __init__(self, kernel, alpha=1e-10, optimizer=None, normalize_y=False,
                 kernel_options={}):
        self.kernel = kernel
        self.alpha = alpha
        self.optimizer = optimizer
        if optimizer is True:
            self.optimizer = 'L-BFGS-B'
        self.normalize_y = normalize_y
        self.kernel_options = kernel_options

    @property
    def X(self):
        try:
            return self._X
        except AttributeError:
            raise AttributeError(
                'Training data does not exist. Please provided using fit().'
            )

    @X.setter
    def X(self, X):
        self._X = np.asarray(X)

    @property
    def y(self):
        try:
            return self._y
        except AttributeError:
            raise AttributeError(
                'Training data does not exist. Please provided using fit().'
            )

    @y.setter
    def y(self, _y):
        if self.normalize_y is True:
            self.y_mean, self.y_std = np.mean(_y), np.std(_y)
            self._y = (np.asarray(_y) - self.y_mean) / self.y_std
        else:
            self.y_mean, self.y_std = 0, 1
            self._y = np.asarray(_y)

    def _gramian(self, X, Y=None, kernel=None, jac=False, diag=False):
        kernel = kernel or self.kernel
        if Y is None:
            if diag is True:
                return kernel.diag(X, **self.kernel_options)
            else:
                if jac is True:
                    K, J = kernel(X, eval_gradient=True, **self.kernel_options)
                    K.flat[::len(K) + 1] += self.alpha
                    return K, J
                else:
                    K = kernel(X, **self.kernel_options)
                    K.flat[::len(K) + 1] += self.alpha
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

    def fit(self, X, y):
        """Train a GPR model. If the `optimizer` argument was set while
        initializing the GPR object, the hyperparameters of the kernel will be
        optimized using maximum likelihood estimation.

        Parameters
        ----------
        X: list of objects or feature vectors.
            Input values of the training data.
        y: 1D array
            Output/target values of the training data.

        Returns
        -------
        self: GaussianProcessRegressor
            returns an instance of self.
        """
        self.X = X
        self.y = y

        '''hyperparameter optimization'''
        if self.optimizer:
            '''maximum likelihood estimation'''
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
                    f'Maximum likelihood estimation did not converge, got:\n'
                    f'{opt}'
                )

        '''build and store GPR model'''
        self.K = K = self._gramian(self.X)
        self.L = CholSolver(K)
        self.Ky = self.L(self.y)
        return self

    def fit_loocv(self, X, y, return_mean=False, return_std=False):
        """Train a GPR model and return the leave-one-out cross validation
        results on the dataset. If the `optimizer` argument was set while
        initializing the GPR object, the hyperparameters of the kernel will be
        optimized with regard to the LOOCV RMSE.

        Parameters
        ----------
        X: list of objects or feature vectors.
            Input values of the training data.
        y: 1D array
            Output/target values of the training data.
        return_mean: boolean
            If True, the leave-one-out predictive mean of the model on the
            training data are returned along with the model.
        return_std: boolean
            If True, the leave-one-out predictive standard deviations of the
            model on the training data are returned along with the model.

        Returns
        -------
        self: GaussianProcessRegressor
            returns an instance of self.
        ymean: 1D array, only if return_mean is True
            Mean of the leave-one-out predictive distribution at query points.
        std: 1D array, only if return_std is True
            Standard deviation of the leave-one-out predictive distribution.
        """
        self.X = X
        self.y = y

        '''hyperparameter optimization'''
        if self.optimizer:
            '''maximum likelihood estimation'''
            opt = minimize(
                fun=lambda theta, self=self: self.squared_loocv_error(
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
                    f'Maximum likelihood estimation did not converge, got:\n'
                    f'{opt}'
                )

        '''build and store GPR model'''
        self.K = self._gramian(X)
        self.L = CholSolver(self.K)
        self.Ky = self.L(y)
        if return_mean is False and return_std is False:
            return self
        else:
            retvals = []
            Kinv_diag = self.L(np.eye(len(self.X))).diagonal()
            if return_mean is True:
                retvals.append(y - self.L(self.y) / Kinv_diag)
            if return_std is True:
                retvals.append(np.sqrt(1 / np.maximum(Kinv_diag, 1e-14)))
            return (self, *retvals)

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
        if not hasattr(self, 'L'):
            raise RuntimeError('Model not trained.')
        Ks = self._gramian(Z, self.X)
        ymean = (Ks @ self.Ky) * self.y_std + self.y_mean
        if return_std is True:
            Kss = self._gramian(Z, diag=True)
            std = np.sqrt(np.maximum(Kss - (Ks @ self.L(Ks.T)).diagonal(), 0))
            return (ymean, std * self.y_std)
        elif return_cov is True:
            Kss = self._gramian(Z)
            cov = np.maximum(Kss - Ks @ self.L(Ks.T), 0)
            return (ymean, cov * self.y_std**2)
        else:
            return ymean

    def predict_loocv(self, Z, z, return_std=False):
        """Compute the leave-one-out cross validation prediction of the given
        data.

        Parameters
        ----------
        Z: list of objects or feature vectors.
            Input values of the unknown data.
        z: 1D array
            Target values of the training data.
        return_std: boolean
            If True, the standard-deviations of the predictions at the query
            points are returned along with the mean.

        Returns
        -------
        ymean: 1D array
            Leave-one-out mean of the predictive distribution at query points.
        std: 1D array
            Leave-one-out standard deviation of the predictive distribution at
            query points.
        """
        assert(len(Z) == len(z))
        z = np.asarray(z)
        if self.normalize_y is True:
            z_mean, z_std = np.mean(z), np.std(z)
            z = (z - z_mean) / z_std
        else:
            z_mean, z_std = 0, 1
        L = CholSolver(self._gramian(Z))
        Kinv_diag = L(np.eye(len(Z))).diagonal()
        ymean = (z - L(z) / Kinv_diag) * z_std + z_mean
        if return_std is True:
            std = np.sqrt(1 / np.maximum(Kinv_diag, 1e-14))
            return (ymean, std * z_std)
        else:
            return ymean

    def log_marginal_likelihood(self, theta=None, X=None, y=None,
                                eval_gradient=False, clone_kernel=True,
                                verbose=False):
        """Returns the log-marginal likelihood of a given set of log-scale
        hyperparameters.

        Parameters
        ----------
        theta: array-like
            Kernel hyperparameters for which the log-marginal likelihood is
            to be evaluated. If None, the current hyperparameters will be used.
        X: list of objects or feature vectors.
            Input values of the training data. If None, the data saved by
            fit() will be used.
        y: 1D array
            Output/target values of the training data. If None, the data saved
            by fit() will be used.
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
        theta = theta if theta is not None else self.kernel.theta
        X = X if X is not None else self.X
        y = y if y is not None else self.y

        if clone_kernel is True:
            kernel = self.kernel.clone_with_theta(theta)
        else:
            kernel = self.kernel
            kernel.theta = theta

        if eval_gradient is True:
            K, dK = self._gramian(X, kernel=kernel, jac=True)
        else:
            K = self._gramian(X, kernel=kernel)

        L = CholSolver(K)
        Ky = L(y)
        yKy = y @ Ky
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

    def squared_loocv_error(self, theta=None, X=None, y=None,
                            eval_gradient=False, clone_kernel=True,
                            verbose=False):
        """Returns the squared LOOCV error of a given set of log-scale
        hyperparameters.

        Parameters
        ----------
        theta: array-like
            Kernel hyperparameters for which the log-marginal likelihood is
            to be evaluated. If None, the current hyperparameters will be used.
        X: list of objects or feature vectors.
            Input values of the training data. If None, the data saved by
            fit() will be used.
        y: 1D array
            Output/target values of the training data. If None, the data saved
            by fit() will be used.
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
        squared_error: float
            Squared LOOCV error of theta for training data.
        squared_error_gradient: 1D array
            Gradient of the Squared LOOCV error with respect to the kernel
            hyperparameters at position theta. Only returned when eval_gradient
            is True.
        """
        theta = theta if theta is not None else self.kernel.theta
        X = X if X is not None else self.X
        y = y if y is not None else self.y

        if clone_kernel is True:
            kernel = self.kernel.clone_with_theta(theta)
        else:
            kernel = self.kernel
            kernel.theta = theta

        if eval_gradient is True:
            K, dK = self._gramian(X, kernel=kernel, jac=True)
        else:
            K = self._gramian(X, kernel=kernel)

        L = CholSolver(K)
        Kinv = L(np.eye(len(X)))
        Kinv_diag = Kinv.diagonal()
        Ky = Kinv @ y
        e = Ky / Kinv_diag
        squared_error = 0.5 * np.sum(e**2)

        if eval_gradient is True:
            D_theta = np.zeros_like(theta)
            for i, t in enumerate(theta):
                dk = dK[:, :, i]
                KdK = Kinv @ dk
                D_theta[i] = (
                    - (e / Kinv_diag) @ (KdK @ Ky)
                    + (e**2 / Kinv_diag) @ (KdK @ Kinv).diagonal()
                ) * np.exp(t)
            return squared_error, D_theta
        else:
            return squared_error
