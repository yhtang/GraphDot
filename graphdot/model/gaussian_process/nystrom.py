#!/usr/bin/env python
# -*- coding: utf-8 -*-
import time
import numpy as np
from graphdot.util.printer import markdown as mprint
import graphdot.linalg.low_rank as lr
from .gpr import GaussianProcessRegressor


class LowRankApproximateGPR(GaussianProcessRegressor):
    r"""Accelerated Gaussian process regression (GPR) using the Nystrom low-rank
    approximation.

    Parameters
    ----------
    kernel: kernel instance
        The covariance function of the GP.
    alpha: float > 0, default = 1e-7
        Value added to the diagonal of the core matrix during fitting. Larger
        values correspond to increased noise level in the observations. A
        practical usage of this parameter is to prevent potential numerical
        stability issues during fitting, and ensures that the core matrix is
        always positive definite in the precense of duplicate entries and/or
        round-off error.
    beta: float > 0, default = 1e-7
        Threshold value for truncating the singular values when computing the
        pseudoinverse of the low-rank kernel matrix. Can be used to tune the
        numerical stability of the model.
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

    In the Nystrom method, the low-rank approximation to $K_{xx}$ is
    \begin{equation}
    \hat{K}_{xx} = K_{xc} K_{cc}^{-1} K_{xc}^T.
    \end{equation}
    However, due to the shear size of $\hat{K}_{xx}$, it is much more
    efficient to compute and store $\hat{K}_{xx}^{-1}$ by
    \begin{equation}
    \begin{aligned}
    \hat{K}_{xx}^{-1}
    &= U_{xx} \Lambda_{xx}^{-1} U_{xx}^T \\
    &= U_{xx} \Lambda_{xx}^{-\frac{1}{2}}
        (U_{xx} \Lambda_{xx}^{-\frac{1}{2}})^T,
    \end{aligned}
    \end{equation}
    where $U_{xx}$ and $\Lambda_{xx}$ are the eigenvectors and eigenvalues
    of $\hat{K}_{xx}$ such that
    $\hat{K}_{xx} = U_{xx} \Lambda_{xx} U_{xx}^T$.
    $U_{xx}$ are also the left singular vectors of
    $\hat{K}_{xx}^\frac{1}{2} \doteq K_{xc} U_{cc} S_{cc}^{-\frac{1}{2}}$,
    while $\Lambda_{xx}^{1/2}$ equal to the singular values $S_{xx}$
    such that
    $\hat{K}_{xx}^\frac{1}{2} = U_{xx} S_{xx} V_{xx}^T$.
    """

    def __init__(self, kernel, alpha=1e-7, beta=1e-7,
                 optimizer=None, normalize_y=False, kernel_options={}):
        self.kernel = kernel
        self.alpha = alpha
        self.beta = beta
        self.optimizer = optimizer
        if optimizer is True:
            self.optimizer = 'L-BFGS-B'
        self.normalize_y = normalize_y
        self.kernel_options = kernel_options

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

    def _set_corespace(self, C, inplace=False):
        r'''
        Given the eigendecomposition $K_{cc} = U_{cc} W_{cc} U_{cc}^T$,
        we have $K_{cc}^{-1} = U_{cc} W_{cc}^{-1} U_{cc}^T$.
        '''
        Kcc = self._gramian(C)
        Kcc.flat[::len(Kcc) + 1] += self.alpha
        a, Q = np.linalg.eigh(Kcc)
        if np.any(a <= 0):
            raise np.linalg.LinAlgError(
                'Core matrix singular, try to increase `alpha`.\n'
            )
        Kcc_rsqrt = Q * a**-0.5
        if inplace is True:
            self.Kcc_rsqrt = Kcc_rsqrt
        return Kcc_rsqrt

    def fit(self, C, X, y, tol=1e-4, repeat=1, theta_jitter=1.0,
            verbose=False):
        """Train a GPR model. If the `optimizer` argument was set while
        initializing the GPR object, the hyperparameters of the kernel will be
        optimized using maximum likelihood estimation.

        Parameters
        ----------
        C: list of objects or feature vectors.
            The core set that defines the subspace of low-rank approximation.
        X: list of objects or feature vectors.
            Input values of the training data.
        y: 1D array
            Output/target values of the training data.
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
            opt = self._hyper_opt(
                lambda theta, self=self: self.log_marginal_likelihood(
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
                    f'Maximum likelihood estimation did not converge, got:\n'
                    f'{opt}'
                )

        '''build and store GPR model'''
        self._set_corespace(self.C, inplace=True)
        self.Kxc = self._gramian(self.X, self.C)
        self.Fxc = self.Kxc @ self.Kcc_rsqrt
        self.Kinv = lr.dot(self.Fxc, rcut=self.beta).inverse()
        self.Ky = self.Kinv @ self.y

        return self

    # def fit_loocv(self, X, y, return_mean=False, return_std=False,
    #               tol=1e-4, repeat=1, theta_jitter=1.0, verbose=False):
    #     """Train a GPR model and return the leave-one-out cross validation
    #     results on the dataset. If the `optimizer` argument was set while
    #     initializing the GPR object, the hyperparameters of the kernel will be
    #     optimized with regard to the LOOCV RMSE.

    #     Parameters
    #     ----------
    #     X: list of objects or feature vectors.
    #         Input values of the training data.
    #     y: 1D array
    #         Output/target values of the training data.
    #     return_mean: boolean
    #         If True, the leave-one-out predictive mean of the model on the
    #         training data are returned along with the model.
    #     return_std: boolean
    #         If True, the leave-one-out predictive standard deviations of the
    #         model on the training data are returned along with the model.
    #     tol: float
    #         Tolerance for termination.
    #     repeat: int
    #         Repeat the hyperparameter optimization by the specified number of
    #         times and return the best result.
    #     theta_jitter: float
    #         Standard deviation of the random noise added to the initial
    #         logscale hyperparameters across repeated optimization runs.
    #     verbose: bool
    #         Whether or not to print out the optimization progress and outcome.

    #     Returns
    #     -------
    #     self: GaussianProcessRegressor
    #         returns an instance of self.
    #     ymean: 1D array, only if return_mean is True
    #         Mean of the leave-one-out predictive distribution at query points.
    #     std: 1D array, only if return_std is True
    #         Standard deviation of the leave-one-out predictive distribution.
    #     """
    #     self.X = X
    #     self.y = y

    #     '''hyperparameter optimization'''
    #     if self.optimizer:
    #         opt = self._hyper_opt(
    #             lambda theta, self=self: self.squared_loocv_error(
    #                 theta, eval_gradient=True, clone_kernel=False,
    #                 verbose=verbose
    #             ),
    #             self.kernel.theta.copy(),
    #             tol, repeat, theta_jitter, verbose
    #         )
    #         if verbose:
    #             print(f'Optimization result:\n{opt}')

    #         if opt.success:
    #             self.kernel.theta = opt.x
    #         else:
    #             raise RuntimeError(
    #                 f'Minimum LOOCV optimization did not converge, got:\n'
    #                 f'{opt}'
    #             )

    #     '''build and store GPR model'''
    #     self._compute_low_rank_subspace(inplace=True)
    #     self.Kxc = self._gramian(self.X, self.C)
    #     self.Fxc = self.Kxc @ self.Kcc_rsqrt
    #     self.Kinv = LowRankApproximateInverse(self.Fxc, self.beta)
    #     self.Ky = self.Kinv @ self.y

    #     if return_mean is False and return_std is False:
    #         return self
    #     else:
    #         retvals = []
    #         Kinv_diag = self.Kinv.diagonal()
    #         if return_mean is True:
    #             ymean = self.y - self.Kinv @ self.y / Kinv_diag
    #             retvals.append(ymean * self.y_std + self.y_mean)
    #         if return_std is True:
    #             ystd = np.sqrt(1 / np.maximum(Kinv_diag, 1e-14))
    #             retvals.append(ystd * self.y_std)
    #         return (self, *retvals)

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
        Kzc = self._gramian(Z, self.C)
        Fzc = Kzc @ self.Kcc_rsqrt
        Kzx = lr.dot(Fzc, self.Fxc.T)

        ymean = Kzx @ self.Ky * self.y_std + self.y_mean
        if return_std is True:
            Kzz = self._gramian(Z, diag=True)
            std = np.sqrt(
                np.maximum(Kzz - (Kzx @ self.Kinv @ Kzx.T).diagonal(), 0)
            )
            return (ymean, std * self.y_std)
        elif return_cov is True:
            Kzz = self._gramian(Z)
            cov = np.maximum(Kzz - (Kzx @ self.Kinv @ Kzx.T).todense(), 0)
            return (ymean, cov * self.y_std**2)
        else:
            return ymean

    # def predict_loocv(self, Z, z, return_std=False):
    #     """Compute the leave-one-out cross validation prediction of the given
    #     data.

    #     Parameters
    #     ----------
    #     Z: list of objects or feature vectors.
    #         Input values of the unknown data.
    #     z: 1D array
    #         Target values of the training data.
    #     return_std: boolean
    #         If True, the standard-deviations of the predictions at the query
    #         points are returned along with the mean.

    #     Returns
    #     -------
    #     ymean: 1D array
    #         Leave-one-out mean of the predictive distribution at query points.
    #     std: 1D array
    #         Leave-one-out standard deviation of the predictive distribution at
    #         query points.
    #     """
    #     assert(len(Z) == len(z))
    #     z = np.asarray(z)
    #     if self.normalize_y is True:
    #         z_mean, z_std = np.mean(z), np.std(z)
    #         z = (z - z_mean) / z_std
    #     else:
    #         z_mean, z_std = 0, 1

    #     if not hasattr(self, 'Kcc_rsqrt'):
    #         self._compute_low_rank_subspace(inplace=True)
    #     Kzc = self._gramian(Z, self.C)

    #     Fzc = Kzc @ self.Kcc_rsqrt

    #     Kinv = LowRankApproximateInverse(Fzc, self.beta)

    #     Kinv_diag = Kinv.diagonal()

    #     mean = (z - Kinv @ z / Kinv_diag) * z_std + z_mean
    #     if return_std is True:
    #         std = np.sqrt(1 / np.maximum(Kinv_diag, 1e-14))
    #         return (mean, std * z_std)
    #     else:
    #         return mean

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
        C = C if C is not None else self.C
        X = X if X is not None else self.X
        y = y if y is not None else self.y

        if clone_kernel is True:
            kernel = self.kernel.clone_with_theta(theta)
        else:
            kernel = self.kernel
            kernel.theta = theta

        t_kernel = time.perf_counter()
        if eval_gradient is True:
            Kxc, d_Kxc = self._gramian(X, C, kernel=kernel, jac=True)
            Kcc, d_Kcc = self._gramian(C, kernel=kernel, jac=True)
        else:
            Kxc = self._gramian(X, C, kernel=kernel)
            Kcc = self._gramian(C, kernel=kernel)
        t_kernel = time.perf_counter() - t_kernel

        t_linalg = time.perf_counter()

        Kcc.flat[::len(C) + 1] += self.alpha
        Acc, Qcc = np.linalg.eigh(Kcc)
        if np.any(Acc <= 0):
            raise np.linalg.LinAlgError(
                'Core matrix singular, try to increase `alpha`.\n'
                f'{Kcc}'
            )
        Kcc_rsqrt = Qcc * Acc**-0.5

        F = Kxc @ Kcc_rsqrt
        K = lr.dot(F, rcut=self.beta)
        K_inv = K.inverse()

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
                ('log(P)', '%12.5g', yKy + logdet),
                ('yKy', '%12.5g', yKy),
                ('logdet(K)', '%12.5g', logdet),
                ('Norm(dK)', '%12.5g', np.linalg.norm(D_theta)),
                ('Cond(K)', '%12.5g', K.cond()),
                ('t_GPU (s)', '%10.2g', t_kernel),
                ('t_CPU (s)', '%10.2g', t_linalg),
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
    #     X = X if X is not None else self.X
    #     y = y if y is not None else self.y

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
