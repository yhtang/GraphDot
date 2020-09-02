#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import numba as nb
import numba.core.types as nbtypes
from graphdot.linalg.cholesky import chol_solve


class VarianceMinimizer:
    '''Select a subset of a dataset such that the Gaussian process posterior
    variance, i.e. the Nystrom residual norm, of the kernel matrix of the
    UNSELECTED samples are as small as possible. In other words,
    the objective here is to ensure that the chosen samples can effectively
    span the vector space as occupied by the entire dataset in a reproducible
    kernel Hilbert space (RKHS).

    Parameters
    ----------
    kernel: callable or 'precomputed'
        A symmetric positive semidefinite function implemented via the
        ``__call__`` semantics. Alternatively, if the value is 'precomputed',
        a square kernel matrix will be expected as an argument to
        :py:`__call__`.
    alpha: float, default=1e-7
        A small value added to the diagonal elements of the kernel matrix in
        order to regularize the variance calculations.
    kernel_options: dict
        Additional arguments to be passed into the kernel.
    '''

    def __init__(self, kernel, alpha=1e-7, kernel_options=None):
        assert kernel == 'precomputed' or callable(kernel)
        self.kernel = kernel
        self.alpha = alpha
        self.kernel_options = kernel_options or {}

    def __call__(self, X, n):
        '''Find a n-sample subset of X that attempts to maximize the diversity
        and return the indices of the samples.

        Parameters
        ----------
        X: feature matrix or list of objects
            Input dataset.
        n: int
            Number of samples to be chosen.

        Returns
        -------
        chosen: list
            Indices of the samples that are chosen.
        '''
        assert len(X) >= n

        if self.kernel == 'precomputed':
            assert (
                isinstance(X, np.ndarray) and
                X.ndim == 2 and
                X.shape[0] == X.shape[1]
            ), 'A precomputed kernel matrix must be square.'
            K = X
        else:
            K = self.kernel(X, **self.kernel_options)

        K.flat[::len(K) + 1] += self.alpha

        return _choose(K.astype(np.float32), n)


@nb.jit(
    nbtypes.intc[:](
        nbtypes.float32[:, :],
        nbtypes.intc
    ),
    forceobj=True
)
def _choose(K, n):
    chosen = np.zeros(len(K), dtype=np.bool_)
    chosen[np.argmax(np.sum(K, axis=1))] = True
    for _ in range(n):
        Ki = K[chosen, :][:, chosen]
        Koi = K[~chosen, :][:, chosen]
        Ko = K[~chosen, :][:, ~chosen]
        var = np.maximum(0, Ko - Koi @ chol_solve(Ki, Koi.T))
        i = np.argmax(np.sum(var, axis=1))
        chosen[np.flatnonzero(~chosen)[i]] = True
    return np.flatnonzero(chosen)
