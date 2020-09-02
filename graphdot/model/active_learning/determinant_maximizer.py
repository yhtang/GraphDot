#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import numba as nb
import numba.core.types as nbtypes


class DeterminantMaximizer:
    '''Select a subset of a dataset such that the determinant of the kernel
    matrix of the selected samples are as large as possible. In other words,
    the objective here is to ensure that the samples are as linearly
    independent as possible in a reproducible kernel Hilbert space (RKHS).

    Parameters
    ----------
    kernel: callable or 'precomputed'
        A symmetric positive semidefinite function implemented via the
        ``__call__`` semantics. Alternatively, if the value is 'precomputed',
        a square kernel matrix will be expected as an argument to
        :py:`__call__`.
    kernel_options: dict
        Additional arguments to be passed into the kernel.
    '''

    def __init__(self, kernel, kernel_options=None):
        assert kernel == 'precomputed' or callable(kernel)
        self.kernel = kernel
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

        return _choose(K.astype(np.float32), n)


@nb.jit(
    nbtypes.intc[:](
        nbtypes.float32[:, :],
        nbtypes.intc
    ),
    forceobj=True
)
def _choose(K, n):
    chosen = []
    for _ in range(n):
        L = np.sum(K**2, axis=1)
        L[chosen] = -np.inf  # ensure chosen points won't be selected again
        i = np.argmax(L)
        chosen.append(i)
        v = K[i, :] / np.linalg.norm(K[i, :])
        K -= np.outer(K @ v, v)
    return chosen
