#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import pickle
import time
import warnings
import numpy as np
from scipy.optimize import minimize
from graphdot.util.printer import markdown as mprint
from graphdot.linalg.cholesky import CholSolver


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
    beta: float > 0, default = 1e-10
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
    kernel_options: dict, optional
        A dictionary of additional options to be passed along when applying the
        kernel to data.
    """

    def __init__(self, kernel, alpha=1e-10, beta=1e-10, optimizer=None,
                 normalize_y=False, kernel_options={}):
        self.kernel = kernel
        self.alpha = alpha
        self.beta = beta
        self.optimizer = optimizer
        if optimizer is True:
            self.optimizer = 'L-BFGS-B'
        self.normalize_y = normalize_y
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
            return self._y
        except AttributeError:
            raise AttributeError(
                'Training data does not exist. Please provide using fit().'
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
                return kernel.diag(X, **self.kernel_options) + self.alpha
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

    def _invert(self, K):
        try:
            return CholSolver(K)
        except np.linalg.LinAlgError:
            try:
                warnings.warn(
                    'Kernel matrix singular, falling back to pseudoinverse'
                )
                return np.linalg.pinv(K, rcond=self.beta, hermitian=True)
            except np.linalg.LinAlgError:
                raise np.linalg.LinAlgError(
                    'The kernel matrix is likely corrupted with NaNs and Infs '
                    'because a pseudoinverse could not be computed.'
                )

    def fit(self, X, y, loss='likelihood', tol=1e-4, repeat=1,
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

            opt = self._hyper_opt(
                lambda theta, objective=objective: objective(
                    theta, eval_gradient=True, clone_kernel=False,
                    verbose=verbose
                ),
                self.kernel.theta.copy(),
                tol, repeat, theta_jitter, verbose
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
        self.K = K = self._gramian(self.X)
        self.Kinv = self._invert(K)
        self.Ky = self.Kinv @ self.y
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
        Ks = self._gramian(Z, self.X)
        ymean = (Ks @ self.Ky) * self.y_std + self.y_mean
        if return_std is True:
            Kss = self._gramian(Z, diag=True)
            std = np.sqrt(
                np.maximum(0, Kss - (Ks @ (self.Kinv @ Ks.T)).diagonal())
            )
            return (ymean, std * self.y_std)
        elif return_cov is True:
            Kss = self._gramian(Z)
            cov = np.maximum(0, Kss - Ks @ (self.Kinv @ Ks.T))
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
        Kinv = self._invert(self._gramian(Z))
        Kinv_diag = (Kinv @ np.eye(len(Z))).diagonal()
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
        X = X if X is not None else self.X
        y = y if y is not None else self.y

        if clone_kernel is True:
            kernel = self.kernel.clone_with_theta(theta)
        else:
            kernel = self.kernel
            kernel.theta = theta

        t_kernel = time.perf_counter()

        if eval_gradient is True:
            K, dK = self._gramian(X, kernel=kernel, jac=True)
        else:
            K = self._gramian(X, kernel=kernel)

        t_kernel = time.perf_counter() - t_kernel

        t_linalg = time.perf_counter()

        Kinv = self._invert(K)
        Ky = Kinv @ y
        yKy = y @ Ky
        logdet = np.prod(np.linalg.slogdet(K))

        if eval_gradient is True:
            d_theta = (
                np.einsum('ij,ijk->k', Kinv @ np.eye(len(K)), dK) -
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
                ('log|K|', '%12.5g', logdet),
                ('Cond(K)', '%12.5g', np.linalg.cond(K)),
                ('t_GPU (s)', '%10.2g', t_kernel),
                ('t_CPU (s)', '%10.2g', t_linalg),
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
        X = X if X is not None else self.X
        y = y if y is not None else self.y

        if clone_kernel is True:
            kernel = self.kernel.clone_with_theta(theta)
        else:
            kernel = self.kernel
            kernel.theta = theta

        t_kernel = time.perf_counter()
        if eval_gradient is True:
            K, dK = self._gramian(X, kernel=kernel, jac=True)
        else:
            K = self._gramian(X, kernel=kernel)
        t_kernel = time.perf_counter() - t_kernel

        t_linalg = time.perf_counter()

        Kinv = self._invert(K)
        if not isinstance(Kinv, np.ndarray):
            Kinv = Kinv @ np.eye(len(X))
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
                ('logdet(K)', '%12.5g', np.prod(np.linalg.slogdet(K))),
                ('Norm(dK)', '%12.5g', np.linalg.norm(D_theta)),
                ('Cond(K)', '%12.5g', np.linalg.cond(K)),
                ('t_GPU (s)', '%10.2g', t_kernel),
                ('t_CPU (s)', '%10.2g', t_linalg),
            )

        return retval

    def _hyper_opt(self, fun, x0, tol, repeat, theta_jitter, verbose):
        X0 = np.copy(self.kernel.theta)
        opt = None

        for r in range(repeat):
            if r > 0:
                x0 = X0 + theta_jitter * np.random.randn(len(X0))
            else:
                x0 = X0

            if verbose:
                mprint.table_start()

            opt_local = minimize(
                fun=fun,
                method=self.optimizer,
                x0=x0,
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
