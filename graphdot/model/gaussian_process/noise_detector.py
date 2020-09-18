#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import pickle
import time
import warnings
import numpy as np
from scipy.optimize import minimize
from graphdot.linalg.cholesky import CholSolver
from graphdot.linalg.spectral import pinvh
from graphdot.util.printer import markdown as mprint
from graphdot.util.iterable import fold_like


class GPRNoiseDetector:
    """Gaussian process regression (GPR) with noise/outlier detection via
    maximum likelihood estimation.

    Parameters
    ----------
    kernel: kernel instance
        The covariance function of the GP.
    alpha_bounds: a tuple of two floats
        Value added to the diagonal of the kernel matrix during fitting. The
        2-tuple will be regarded as the lower and upper bounds of the
        values added to each diagonal element, which will be
        optimized individually by training.
        Larger values correspond to increased noise level in the observations.
        A practical usage of this parameter is to prevent potential numerical
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
    kernel_options: dict, optional
        A dictionary of additional options to be passed along when applying the
        kernel to data.
    """

    def __init__(self, kernel, alpha_bounds=(1e-8, np.inf), beta=1e-8,
                 optimizer=True, normalize_y=False, kernel_options={}):
        self.kernel = kernel
        self.sigma_bounds = np.sqrt(alpha_bounds)
        self.beta = beta
        self.optimizer = optimizer
        if self.optimizer is True:
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

    @property
    def y_uncertainty(self):
        '''The learned uncertainty magnitude of each training sample.'''
        try:
            return self._sigma
        except AttributeError:
            raise AttributeError(
                'Uncertainty must be learned via fit().'
            )

    def _gramian(self, alpha, X, Y=None, kernel=None, jac=False, diag=False):
        kernel = kernel or self.kernel
        if Y is None:
            if diag is True:
                return kernel.diag(X, **self.kernel_options) + alpha
            else:
                if jac is True:
                    K, J = kernel(X, eval_gradient=True, **self.kernel_options)
                    K.flat[::len(K) + 1] += alpha
                    return K, J
                else:
                    K = kernel(X, **self.kernel_options)
                    K.flat[::len(K) + 1] += alpha
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
            return CholSolver(K), np.prod(np.linalg.slogdet(K))
        except np.linalg.LinAlgError:
            try:
                warnings.warn(
                    'Kernel matrix singular, falling back to pseudoinverse'
                )
                return pinvh(
                    K, rcond=self.beta, mode='clamp', return_nlogdet=True
                )
            except np.linalg.LinAlgError:
                raise np.linalg.LinAlgError(
                    'The kernel matrix is likely corrupted with NaNs and Infs '
                    'because a pseudoinverse could not be computed.'
                )

    def fit(self, X, y, w, udistro=None, tol=1e-5, repeat=1,
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
        w: float
            The strength of L1 penalty on the noise terms.
        udistro: callable
            A random number generator for the initial guesses of the
            uncertainties. A lognormal distribution will be used by
            default if the argument is None.
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

            opt = self._hyper_opt_l1reg(
                lambda theta_ext: self.log_marginal_likelihood(
                    theta_ext, eval_gradient=True, clone_kernel=False,
                    verbose=verbose
                ),
                self.kernel.theta,
                udistro,
                w, tol, repeat, theta_jitter, verbose
            )
            if verbose:
                print(f'Optimization result:\n{opt}')

            if opt.success:
                self.kernel.theta, log_sigma = fold_like(
                    opt.x,
                    (self.kernel.theta, self.y)
                )
                self._sigma = np.exp(log_sigma)
            else:
                raise RuntimeError(
                    f'Training did not converge, got:\n'
                    f'{opt}'
                )

        '''build and store GPR model'''
        self.K = K = self._gramian(self._sigma**2, self.X)
        self.Kinv, _ = self._invert(K)
        self.Ky = self.Kinv @ self.y
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
        if not hasattr(self, 'Kinv'):
            raise RuntimeError('Model not trained.')
        Ks = self._gramian(0, Z, self.X)
        ymean = (Ks @ self.Ky) * self.y_std + self.y_mean
        if return_std is True:
            Kss = self._gramian(0, Z, diag=True)
            std = np.sqrt(
                np.maximum(0, Kss - (Ks @ (self.Kinv @ Ks.T)).diagonal())
            )
            return (ymean, std * self.y_std)
        elif return_cov is True:
            Kss = self._gramian(0, Z)
            cov = np.maximum(0, Kss - Ks @ (self.Kinv @ Ks.T))
            return (ymean, cov * self.y_std**2)
        else:
            return ymean

    def log_marginal_likelihood(self, theta_ext, X=None, y=None,
                                eval_gradient=False, clone_kernel=True,
                                verbose=False):
        """Returns the log-marginal likelihood of a given set of log-scale
        hyperparameters.

        Parameters
        ----------
        theta_ext: array-like
            Kernel hyperparameters and per-sample noise prior for which the
            log-marginal likelihood is to be evaluated. If None, the current
            hyperparameters will be used.
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
        X = X if X is not None else self.X
        y = y if y is not None else self.y
        theta, log_sigma = fold_like(theta_ext, (self.kernel.theta, y))
        sigma = np.exp(log_sigma)

        if clone_kernel is True:
            kernel = self.kernel.clone_with_theta(theta)
        else:
            kernel = self.kernel
            kernel.theta = theta

        t_kernel = time.perf_counter()

        if eval_gradient is True:
            K, dK = self._gramian(sigma**2, X, kernel=kernel, jac=True)
        else:
            K = self._gramian(sigma**2, X, kernel=kernel)

        t_kernel = time.perf_counter() - t_kernel

        t_linalg = time.perf_counter()

        Kinv, logdet = self._invert(K)
        try:
            Kinv_diag = Kinv.diagonal()
        except AttributeError:
            Kinv = Kinv @ np.eye(len(y))
            Kinv_diag = Kinv.diagonal()
        Ky = Kinv @ y
        yKy = y @ Ky

        if eval_gradient is True:
            d_theta = (
                np.einsum('ij,ijk->k', Kinv, dK) -
                np.einsum('i,ijk,j', Ky, dK, Ky)
            )
            d_alpha = (Kinv_diag - Ky**2) * 2 * sigma
            retval = (
                yKy + logdet,
                np.concatenate((d_theta, d_alpha)) * np.exp(theta_ext)
            )
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

    def _hyper_opt_l1reg(
        self, fun, theta0, udistro, w, tol, repeat, theta_jitter, verbose
    ):
        if udistro is None:
            def udistro(n):
                return np.std(self.y) * np.random.lognormal(-1.0, 1.0, n)
        assert callable(udistro)

        penalty = np.concatenate((
            np.zeros_like(theta0),
            np.ones_like(self.y) * w
        ))

        def ext_fun(x):
            exp_x = np.exp(x)
            val, jac = fun(x)
            return (
                val + np.linalg.norm(penalty * exp_x, ord=1),
                jac + penalty * exp_x
            )

        opt = None

        for r in range(repeat):
            if verbose:
                mprint.table_start()

            if r == 0:
                theta = theta0
            else:
                theta = theta0 + theta_jitter * np.random.randn(len(theta))

            opt_local = minimize(
                fun=ext_fun,
                method=self.optimizer,
                x0=np.concatenate((theta, np.log(udistro(len(self.y))))),
                bounds=np.vstack((
                    self.kernel.bounds,
                    np.tile(np.log(self.sigma_bounds), (len(self.y), 1)),
                )),
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
