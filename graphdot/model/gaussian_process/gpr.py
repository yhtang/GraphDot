#!/usr/bin/env python
# -*- coding: utf-8 -*-
import time
import numpy as np
from graphdot.util.printer import markdown as mprint
from .base import GaussianProcessRegressorBase


class GaussianProcessRegressor(GaussianProcessRegressorBase):
    """Gaussian process regression (GPR).

    Parameters
    ----------
    kernel: kernel instance
        The covariance function of the GP.
    alpha: float > 0
        Value added to the diagonal of the kernel matrix during fitting. Larger
        values correspond to increased noise level in the observations. A
        practical usage of this parameter is to prevent potential numerical
        stability issues during fitting, and ensures that the kernel matrix is
        always positive definite in the precense of duplicate entries and/or
        round-off error.
    beta: float > 0
        Cutoff value on the singular values for the spectral pseudoinverse
        computation, which serves as a backup mechanism to invert the kernel
        matrix in case if it is singular.
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
    regularization: '+' or 'additive' or '*' or 'multiplicative'
        Determines the method of regularization. If '+' or 'additive',
        ``alpha`` is added to the diagonals of the kernel matrix. If '*' or
        'multiplicative', a factor of ``1 + alpha`` will be multiplied with
        each diagonal element.
    kernel_options: dict, optional
        A dictionary of additional options to be passed along when applying the
        kernel to data.
    """

    def __init__(self, kernel, alpha=1e-8, beta=1e-8, optimizer=None,
                 normalize_y=False, regularization='+',
                 kernel_options={}):
        super().__init__(
            kernel,
            normalize_y=normalize_y,
            regularization=regularization,
            kernel_options=kernel_options
        )
        self.alpha = alpha
        self.beta = beta
        self.optimizer = optimizer
        if optimizer is True:
            self.optimizer = 'L-BFGS-B'

    def fit(self, X, y, loss='likelihood', tol=1e-5, repeat=1,
            theta_jitter=1.0, verbose=False):
        """Train a GPR model. If the `optimizer` argument was set while
        initializing the GPR object, the hyperparameters of the kernel will be
        optimized using the specified loss function.

        Parameters
        ----------
        X: list of objects or feature vectors.
            Input values of the training data.
        y: 1D array
            Output/target values of the training data.
        loss: 'likelihood' or 'loocv'
            The loss function to be minimzed during training. Could be either
            'likelihood' (negative log-likelihood) or 'loocv' (mean-square
            leave-one-out cross validation error).
        tol: float
            Tolerance for termination.
        repeat: int
            Repeat the hyperparameter optimization by the specified number of
            times and return the best result.
        theta_jitter: float
            Standard deviation of the random noise added to the initial
            logscale hyperparameters across repeated optimization runs.
        verbose: bool
            Whether or not to print out the optimization progress and outcome.

        Returns
        -------
        self: GaussianProcessRegressor
            returns an instance of self.
        """
        self.X = X
        self.y = y

        '''hyperparameter optimization'''
        if self.optimizer:

            if loss == 'likelihood':
                objective = self.log_marginal_likelihood
            elif loss == 'loocv':
                objective = self.squared_loocv_error
            else:
                raise RuntimeError(f'Unknown loss function: {loss}.')

            def xgen(n):
                x0 = self.kernel.theta.copy()
                yield x0
                yield from x0 + theta_jitter * np.random.randn(n - 1, len(x0))

            opt = self._hyper_opt(
                method=self.optimizer,
                fun=lambda theta, objective=objective: objective(
                    theta, eval_gradient=True, clone_kernel=False,
                    verbose=verbose
                ),
                xgen=xgen(repeat), tol=tol, verbose=verbose
            )
            if verbose:
                print(f'Optimization result:\n{opt}')

            if opt.success:
                self.kernel.theta = opt.x
            else:
                raise RuntimeError(
                    f'Training using the {loss} loss did not converge, got:\n'
                    f'{opt}'
                )

        '''build and store GPR model'''
        K = self._gramian(self.alpha, self._X)
        self.K = K = K[self._y_mask, :][:, self._y_mask]
        self.Kinv, _ = self._invert(K, rcond=self.beta)
        self.Ky = self.Kinv @ self._y
        return self

    def fit_loocv(self, X, y, **options):
        """Alias of :py:`fit_loocv(X, y, loss='loocv', **options)`."""
        return self.fit(X, y, loss='loocv', **options)

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
        if not hasattr(self, 'Kinv'):
            raise RuntimeError('Model not trained.')
        Ks = self._gramian(None, Z, self._X)[:, self._y_mask]
        ymean = (Ks @ self.Ky) * self._ystd + self._ymean
        if return_std is True:
            Kss = self._gramian(self.alpha, Z, diag=True)
            std = np.sqrt(
                np.maximum(0, Kss - (Ks @ (self.Kinv @ Ks.T)).diagonal())
            )
            return (ymean, std * self._ystd)
        elif return_cov is True:
            Kss = self._gramian(self.alpha, Z)
            cov = np.maximum(0, Kss - Ks @ (self.Kinv @ Ks.T))
            return (ymean, cov * self._ystd**2)
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
        z_mask, z_masked = self.mask(z)
        if self.normalize_y is True:
            z_mean, z_std = np.mean(z_masked), np.std(z_masked)
            z = (z_masked - z_mean) / z_std
        else:
            z_mean, z_std = 0, 1
            z = z_masked

        K = self._gramian(self.alpha, Z)[z_mask, :][:, z_mask]
        Kinv, _ = self._invert(K, rcond=self.beta)
        Kinv_diag = Kinv.diagonal()
        ymean = (z - Kinv @ z / Kinv_diag) * z_std + z_mean
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
            Input values of the training data. If None, `self.X` will be used.
        y: 1D array
            Output/target values of the training data. If None, `self.y` will
            be used.
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
        X = X if X is not None else self._X
        if y is not None:
            y_mask, y = self.mask(y)
        else:
            y = self._y
            y_mask = self._y_mask

        if clone_kernel is True:
            kernel = self.kernel.clone_with_theta(theta)
        else:
            kernel = self.kernel
            kernel.theta = theta

        t_kernel = time.perf_counter()

        if eval_gradient is True:
            K, dK = self._gramian(self.alpha, X, kernel=kernel, jac=True)
            K = K[y_mask, :][:, y_mask]
            dK = dK[y_mask, :, :][:, y_mask, :]
        else:
            K = self._gramian(self.alpha, X, kernel=kernel)
            K = K[y_mask, :][:, y_mask]

        t_kernel = time.perf_counter() - t_kernel

        t_linalg = time.perf_counter()

        Kinv, logdet = self._invert(K, rcond=self.beta)
        Ky = Kinv @ y
        yKy = y @ Ky

        if eval_gradient is True:
            if not isinstance(Kinv, np.ndarray):
                Kinv = Kinv.todense()
            d_theta = (
                np.einsum('ij,ijk->k', Kinv, dK) -
                np.einsum('i,ijk,j', Ky, dK, Ky)
            )
            retval = (yKy + logdet, d_theta * np.exp(theta))
        else:
            retval = yKy + logdet

        t_linalg = time.perf_counter() - t_linalg

        if verbose:
            mprint.table(
                ('logP', '%12.5g', yKy + logdet),
                ('dlogP', '%12.5g', np.linalg.norm(d_theta)),
                ('y^T.K.y', '%12.5g', yKy),
                ('log|K| ', '%12.5g', logdet),
                ('Cond(K)', '%12.5g', np.linalg.cond(K)),
                ('GPU time', '%10.2g', t_kernel),
                ('CPU time', '%10.2g', t_linalg),
            )

        return retval

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
            Input values of the training data. If None, `self.X` will be used.
        y: 1D array
            Output/target values of the training data. If None, `self.y` will
            be used.
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
        X = X if X is not None else self._X
        if y is not None:
            y_mask, y = self.mask(y)
        else:
            y = self._y
            y_mask = self._y_mask

        if clone_kernel is True:
            kernel = self.kernel.clone_with_theta(theta)
        else:
            kernel = self.kernel
            kernel.theta = theta

        t_kernel = time.perf_counter()

        if eval_gradient is True:
            K, dK = self._gramian(self.alpha, X, kernel=kernel, jac=True)
            K = K[y_mask, :][:, y_mask]
            dK = dK[y_mask, :, :][:, y_mask, :]
        else:
            K = self._gramian(self.alpha, X, kernel=kernel)
            K = K[y_mask, :][:, y_mask]

        t_kernel = time.perf_counter() - t_kernel

        t_linalg = time.perf_counter()

        Kinv, logdet = self._invert(K, rcond=self.beta)
        if not isinstance(Kinv, np.ndarray):
            Kinv = Kinv.todense()
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
            retval = (squared_error, D_theta)
        else:
            retval = squared_error

        t_linalg = time.perf_counter() - t_linalg

        if verbose:
            mprint.table(
                ('Sq.Err.', '%12.5g', squared_error),
                ('d(SqErr)', '%12.5g', squared_error),
                ('log|K| ', '%12.5g', logdet),
                ('Cond(K)', '%12.5g', np.linalg.cond(K)),
                ('t_GPU (s)', '%10.2g', t_kernel),
                ('t_CPU (s)', '%10.2g', t_linalg),
            )

        return retval
