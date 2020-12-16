#!/usr/bin/env python
# -*- coding: utf-8 -*-
import time
import warnings
import numpy as np
from graphdot.util.printer import markdown as mprint
import graphdot.linalg.low_rank as lr
from graphdot.linalg.spectral import powerh
from .base import GaussianProcessRegressorBase


class LowRankApproximateGPR(GaussianProcessRegressorBase):
    r"""Accelerated Gaussian process regression (GPR) using the Nystrom low-rank
    approximation.

    Parameters
    ----------
    kernel: kernel instance
        The covariance function of the GP.
    alpha: float > 0
        Value added to the diagonal of the core matrix during fitting. Larger
        values correspond to increased noise level in the observations. A
        practical usage of this parameter is to prevent potential numerical
        stability issues during fitting, and ensures that the core matrix is
        always positive definite in the precense of duplicate entries and/or
        round-off error.
    beta: float > 0
        Cutoff value on the singular values for the spectral pseudoinverse
        of the low-rank kernel matrix.
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

    def __init__(self, kernel, alpha=1e-7, beta=1e-7, optimizer=None,
                 normalize_y=False, regularization='+', kernel_options={}):
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

    @property
    def C(self):
        '''The core sample set for constructing the subspace for low-rank
        approximation.'''
        try:
            return self._C
        except AttributeError:
            raise AttributeError(
                'Core samples do not exist. Please provide using fit().'
            )

    @C.setter
    def C(self, C):
        self._C = C

    def _corespace(self, C=None, Kcc=None):
        assert(C is None or Kcc is None)
        if Kcc is None:
            Kcc = self._gramian(self.alpha, C)
        try:
            return powerh(Kcc, -0.5, return_symmetric=False)
        except np.linalg.LinAlgError:
            warnings.warn(
                'Core matrix singular, try to increase `alpha`.\n'
                'Now falling back to use a pseudoinverse.'
            )
            try:
                return powerh(Kcc, -0.5, rcond=self.beta, mode='clamp',
                              return_symmetric=False)
            except np.linalg.LinAlgError:
                raise np.linalg.LinAlgError(
                    'The core matrix is likely corrupted with NaNs and Infs '
                    'because a pseudoinverse could not be computed.'
                )

    def fit(self, C, X, y, loss='likelihood', tol=1e-5, repeat=1,
            theta_jitter=1.0, verbose=False):
        """Train a low-rank approximate GPR model. If the `optimizer` argument
        was set while initializing the GPR object, the hyperparameters of the
        kernel will be optimized using the specified loss function.

        Parameters
        ----------
        C: list of objects or feature vectors.
            The core set that defines the subspace of low-rank approximation.
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
        self: LowRankApproximateGPR
            returns an instance of self.
        """
        self.C = C
        self.X = X
        self.y = y

        '''hyperparameter optimization'''
        if self.optimizer:

            if loss == 'likelihood':
                objective = self.log_marginal_likelihood
            elif loss == 'loocv':
                raise NotImplementedError(
                    '(ง๑ •̀_•́)ง LOOCV training not ready yet.'
                )

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
        self.Kcc_rsqrt = self._corespace(C=self._C)
        self.Kxc = self._gramian(None, self._X, self._C)[self._y_mask, :]
        self.Fxc = self.Kxc @ self.Kcc_rsqrt
        self.Kinv = lr.dot(self.Fxc, rcond=self.beta, mode='clamp').pinv()
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
        Kzc = self._gramian(None, Z, self._C)
        Fzc = Kzc @ self.Kcc_rsqrt
        Kzx = lr.dot(Fzc, self.Fxc.T)

        ymean = Kzx @ self.Ky * self._ystd + self._ymean
        if return_std is True:
            Kzz = self._gramian(self.alpha, Z, diag=True)
            std = np.sqrt(
                np.maximum(Kzz - (Kzx @ self.Kinv @ Kzx.T).diagonal(), 0)
            )
            return (ymean, std * self._ystd)
        elif return_cov is True:
            Kzz = self._gramian(self.alpha, Z)
            cov = np.maximum(Kzz - (Kzx @ self.Kinv @ Kzx.T).todense(), 0)
            return (ymean, cov * self._ystd**2)
        else:
            return ymean

    def predict_loocv(self, Z, z, return_std=False, method='auto'):
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
        method: 'auto' or 'ridge-like' or 'gpr-like'
            Selects the algorithm used for fast evaluation of the leave-one-out
            cross validation without expliciting training one model per sample.
            'ridge-like' seems to be more stable with a smaller core size (that
            is not rank-deficit), while 'gpr-like' seems to be more stable with
            a larger core size. By default, the option is 'auto' and the
            function will choose a method based on an analysis on the
            eigenspectrum of the dataset.

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

        if not hasattr(self, 'Kcc_rsqrt'):
            raise RuntimeError('Model not trained.')
        Kzc = self._gramian(None, Z, self._C)

        Cov = Kzc.T @ Kzc
        Cov.flat[::len(self._C) + 1] += self.alpha
        Cov_rsqrt, eigvals = powerh(
            Cov, -0.5, return_symmetric=False, return_eigvals=True
        )

        # if an eigenvalue is smaller than alpha, it would have been negative
        # in the unregularized Cov matrix
        if method == 'auto':
            if eigvals.min() > self.alpha:
                method = 'ridge-like'
            else:
                method = 'gpr-like'

        if method == 'ridge-like':
            P = Kzc @ Cov_rsqrt
            L = lr.dot(P, P.T)
            zstar = z - (z - L @ z) / (1 - L.diagonal())
            if return_std is True:
                raise NotImplementedError(
                    'LOOCV std using the ridge-like method is not ready yet.'
                )
        elif method == 'gpr-like':
            F = Kzc @ self.Kcc_rsqrt
            Kinv = lr.dot(F, rcond=self.beta, mode='clamp').pinv()
            zstar = z - (Kinv @ z) / Kinv.diagonal()
            if return_std is True:
                std = np.sqrt(1 / np.maximum(Kinv.diagonal(), 1e-14))
        else:
            raise RuntimeError(f'Unknown method {method} for predict_loocv.')

        if return_std is True:
            return (zstar * z_std + z_mean, std * z_std)
        else:
            return zstar * z_std + z_mean

    def log_marginal_likelihood(self, theta=None, C=None, X=None, y=None,
                                eval_gradient=False, clone_kernel=True,
                                verbose=False):
        """Returns the log-marginal likelihood of a given set of log-scale
        hyperparameters.

        Parameters
        ----------
        theta: array-like
            Kernel hyperparameters for which the log-marginal likelihood is
            to be evaluated. If None, the current hyperparameters will be used.
        C: list of objects or feature vectors.
            The core set that defines the subspace of low-rank approximation.
            If None, `self.C` will be used.
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
        C = C if C is not None else self._C
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
            Kxc, d_Kxc = self._gramian(None, X, C, kernel=kernel, jac=True)
            Kcc, d_Kcc = self._gramian(self.alpha, C, kernel=kernel, jac=True)
            Kxc, d_Kxc = Kxc[y_mask, :], d_Kxc[y_mask, :, :]
        else:
            Kxc = self._gramian(None, X, C, kernel=kernel)
            Kcc = self._gramian(self.alpha, C, kernel=kernel)
            Kxc = Kxc[y_mask, :]

        t_kernel = time.perf_counter() - t_kernel

        t_linalg = time.perf_counter()

        Kcc_rsqrt = self._corespace(Kcc=Kcc)
        F = Kxc @ Kcc_rsqrt
        K = lr.dot(F, rcond=self.beta, mode='clamp')
        K_inv = K.pinv()

        logdet = K.logdet()
        Ky = K_inv @ y
        yKy = y @ Ky
        logP = yKy + logdet

        if eval_gradient is True:
            D_theta = np.zeros_like(theta)
            K_inv2 = K_inv**2
            for i, t in enumerate(theta):
                d_F = d_Kxc[:, :, i] @ Kcc_rsqrt
                d_K = lr.dot(F, d_F.T) + lr.dot(d_F, F.T) - lr.dot(
                            F @ Kcc_rsqrt.T @ d_Kcc[:, :, i],
                            Kcc_rsqrt @ F.T
                        )
                d_logdet = (K_inv @ d_K).trace()
                d_Kinv_part = K_inv2 @ d_K - K_inv2 @ d_K @ (K @ K_inv)
                d_Kinv = d_Kinv_part + d_Kinv_part.T - K_inv @ d_K @ K_inv
                d_yKy = d_Kinv.quadratic(y, y)
                D_theta[i] = (d_logdet + d_yKy) * np.exp(t)
            retval = (logP, D_theta)
        else:
            retval = logP

        t_linalg = time.perf_counter() - t_linalg

        if verbose:
            mprint.table(
                ('logP', '%12.5g', yKy + logdet),
                ('dlogP', '%12.5g', np.linalg.norm(D_theta)),
                ('y^T.K.y', '%12.5g', yKy),
                ('log|K| ', '%12.5g', logdet),
                ('Cond(K)', '%12.5g', K.cond()),
                ('GPU time', '%10.2g', t_kernel),
                ('CPU time', '%10.2g', t_linalg),
            )

        return retval

    # def squared_loocv_error(self, theta=None, X=None, y=None,
    #                         eval_gradient=False, clone_kernel=True,
    #                         verbose=False):
    #     """Returns the squared LOOCV error of a given set of log-scale
    #     hyperparameters.

    #     Parameters
    #     ----------
    #     theta: array-like
    #         Kernel hyperparameters for which the log-marginal likelihood is
    #         to be evaluated. If None, the current hyperparameters will be
    #         used.
    #     X: list of objects or feature vectors.
    #         Input values of the training data. If None, the data saved by
    #         fit() will be used.
    #     y: 1D array
    #         Output/target values of the training data. If None, the data
    #         saved by fit() will be used.
    #     eval_gradient: boolean
    #         If True, the gradient of the log-marginal likelihood with respect
    #         to the kernel hyperparameters at position theta will be returned
    #         alongside.
    #     clone_kernel: boolean
    #         If True, the kernel is copied so that probing with theta does not
    #         alter the trained kernel. If False, the kernel hyperparameters
    #         will be modified in-place.
    #     verbose: boolean
    #         If True, the log-likelihood value and its components will be
    #         printed to the screen.

    #     Returns
    #     -------
    #     squared_error: float
    #         Squared LOOCV error of theta for training data.
    #     squared_error_gradient: 1D array
    #         Gradient of the Squared LOOCV error with respect to the kernel
    #         hyperparameters at position theta. Only returned when
    #         eval_gradient is True.
    #     """
    #     raise RuntimeError('Not implemented')
    #     theta = theta if theta is not None else self.kernel.theta
    #     X = X if X is not None else self._X
    #     y = y if y is not None else self._y

    #     if clone_kernel is True:
    #         kernel = self.kernel.clone_with_theta(theta)
    #     else:
    #         kernel = self.kernel
    #         kernel.theta = theta

    #     t_kernel = time.perf_counter()
    #     if eval_gradient is True:
    #         K, dK = self._gramian(X, kernel=kernel, jac=True)
    #     else:
    #         K = self._gramian(X, kernel=kernel)
    #     t_kernel = time.perf_counter() - t_kernel

    #     t_linalg = time.perf_counter()
    #     L = CholSolver(K)
    #     Kinv = L(np.eye(len(X)))
    #     Kinv_diag = Kinv.diagonal()
    #     Ky = Kinv @ y
    #     e = Ky / Kinv_diag
    #     squared_error = 0.5 * np.sum(e**2)
    #     t_linalg = time.perf_counter() - t_linalg

    #     if eval_gradient is True:
    #         D_theta = np.zeros_like(theta)
    #         for i, t in enumerate(theta):
    #             dk = dK[:, :, i]
    #             KdK = Kinv @ dk
    #             D_theta[i] = (
    #                 - (e / Kinv_diag) @ (KdK @ Ky)
    #                 + (e**2 / Kinv_diag) @ (KdK @ Kinv).diagonal()
    #             ) * np.exp(t)
    #         if verbose:
    #             mprint.table(
    #                 ('Sq.Err.', '%12.5g', squared_error),
    #                 ('logdet(K)', '%12.5g', np.prod(np.linalg.slogdet(K))),
    #                 ('Norm(dK)', '%12.5g', np.linalg.norm(D_theta)),
    #                 ('Cond(K)', '%12.5g', np.linalg.cond(K)),
    #                 ('t_GPU (s)', '%10.2g', t_kernel),
    #                 ('t_CPU (s)', '%10.2g', t_linalg),
    #             )
    #         return squared_error, D_theta
    #     else:
    #         return squared_error
