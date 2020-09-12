#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from graphdot.linalg.block import binvh1


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

    def __init__(self, kernel, alpha=1e-6, kernel_options=None):
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
            K = np.copy(X).astype(np.float)
        else:
            K = self.kernel(X, **self.kernel_options).astype(np.float)

        K.flat[::len(K) + 1] += self.alpha

        return self._choose(K, n)

    @staticmethod
    def _choose(K, n):
        chosen = []
        index = np.arange(len(K))
        inv = np.zeros((0, 0))
        for i in range(n):
            posterior = K[i:, i:] - K[i:, :i] @ inv @ K[:i, i:]
            j = i + np.argmax(np.sum(posterior, axis=1))
            chosen.append(index[j])
            index[[i, j]] = index[[j, i]]
            K[[i, j], :] = K[[j, i], :]
            K[:, [i, j]] = K[:, [j, i]]
            if i < n - 1:
                inv = binvh1(inv, K[:i, i], K[i, i])
        return chosen
