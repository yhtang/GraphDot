#!/usr/bin/env python
# -*- coding: utf-8 -*-
import time
import numpy as np
from scipy.optimize import minimize
from graphdot.util.printer import markdown as mprint
from graphdot.util.iterable import fold_like
from .base import GaussianProcessRegressorBase


class GPROutlierDetector(GaussianProcessRegressorBase):
    """Gaussian process regression (GPR) with noise/outlier detection via
    maximum likelihood estimation.

    Parameters
    ----------
    kernel: kernel instance
        The covariance function of the GP.
    sigma_bounds: a tuple of two floats
        As Value added to the diagonal of the kernel matrix during fitting. The
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

    def __init__(self, kernel, sigma_bounds=(1e-4, np.inf), beta=1e-8,
                 optimizer=True, normalize_y=False, kernel_options={}):
        super().__init__(
            kernel,
            normalize_y=normalize_y,
            kernel_options=kernel_options,
            regularization='+'
        )
        self.sigma_bounds = sigma_bounds
        self.beta = beta
        self.optimizer = optimizer
        if self.optimizer is True:
            self.optimizer = 'L-BFGS-B'

    @property
    def y_uncertainty(self):
        '''The learned uncertainty magnitude of each training sample.'''
        try:
            return self._sigma * self._ystd
        except AttributeError:
            raise AttributeError(
                'Uncertainty must be learned via fit().'
            )

    def fit(self, X, y, w, udist=None, tol=1e-4, repeat=1,
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
        udist: callable
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

            def xgen(n):
                x0 = self.kernel.theta.copy()
                yield x0
                yield from x0 + theta_jitter * np.random.randn(n - 1, len(x0))

            opt = self._hyper_opt_l1reg(
                method=self.optimizer,
                fun=lambda theta_ext: self.log_marginal_likelihood(
                    theta_ext, eval_gradient=True, clone_kernel=False,
                    verbose=verbose
                ),
                xgen=xgen(repeat),
                udist=udist, w=w, tol=tol, verbose=verbose
            )
            if verbose:
                print(f'Optimization result:\n{opt}')

            if opt.success:
                self.kernel.theta, log_sigma = fold_like(
                    opt.x,
                    (self.kernel.theta, self._y)
                )
                self._sigma = np.exp(log_sigma)
            else:
                raise RuntimeError(
                    f'Training did not converge, got:\n'
                    f'{opt}'
                )

        '''build and store GPR model'''
        self.K = K = self._gramian(self._sigma**2, self._X)
        self.Kinv, _ = self._invert_pseudoinverse(K, rcond=self.beta)
        self.Ky = self.Kinv @ self._y
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
        Ks = self._gramian(None, Z, self._X)
        ymean = (Ks @ self.Ky) * self._ystd + self._ymean
        if return_std is True:
            Kss = self._gramian(0, Z, diag=True)
            std = np.sqrt(
                np.maximum(0, Kss - (Ks @ (self.Kinv @ Ks.T)).diagonal())
            )
            return (ymean, std * self._ystd)
        elif return_cov is True:
            Kss = self._gramian(0, Z)
            cov = np.maximum(0, Kss - Ks @ (self.Kinv @ Ks.T))
            return (ymean, cov * self._ystd**2)
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
        X = X if X is not None else self._X
        y = y if y is not None else self._y
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

        Kinv, logdet = self._invert_pseudoinverse(K, rcond=self.beta)
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
        self, method, fun, xgen, udist, w, tol, verbose
    ):
        if udist is None:
            def udist(n):
                return self._ystd * np.random.lognormal(-1.0, 1.0, n)
        assert callable(udist)

        penalty = np.concatenate((
            np.zeros_like(self.kernel.theta),
            np.ones_like(self._y) * w
        ))

        def ext_fun(x):
            exp_x = np.exp(x)
            val, jac = fun(x)
            return (
                val + np.linalg.norm(penalty * exp_x, ord=1),
                jac + penalty * exp_x
            )

        opt = None

        for x in xgen:
            if verbose:
                mprint.table_start()

            opt_local = minimize(
                fun=ext_fun,
                method=self.optimizer,
                x0=np.concatenate((x, np.log(udist(len(self._y))))),
                bounds=np.vstack((
                    self.kernel.bounds,
                    np.tile(np.log(self.sigma_bounds), (len(self._y), 1)),
                )),
                jac=True,
                tol=tol,
            )

            if not opt or (opt_local.success and opt_local.fun < opt.fun):
                opt = opt_local

        return opt
