#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import numba as nb
import numba.core.types as nbtypes


class StochasticVolumeMaximizer:
    '''Select a subset of a dataset such that the samples are as linearly
    independent as possible in a reproducible kernel Hilbert space (RKHS).

    Parameters
    ----------
    kernel: callable
        A symmetric positive semidefinite function that implements the
        :py:`__call__` semantics.
    k: int > 1
        The branching factor of the search algorithm.
    batch: int
        The size of mini-batches as used by the search algorithm.
    kernel_options: dict
        Additional arguments to be passed into the kernel.
    '''

    def __init__(self, kernel, k=2, batch='auto', kernel_options=None):
        assert k > 1, "k must be an integer greater than 1"
        self.kernel = kernel
        self.k = k
        self.batch = batch
        self.kernel_options = kernel_options or {}

    def __call__(self, X, n, random_state=None):
        '''Find a n-sample subset of X that attempts to maximize the diversity
        and return the indices of the samples.

        Parameters
        ----------
        X: feature matrix or list of objects
            Input dataset.
        n: int
            The size of the subset to be chosen.
        random_state: int or :py:`np.random.Generator`
            The seed to the random number generator (RNG), or the RNG itself.
            If None, the default RNG in numpy will be used.

        Returns
        -------
        chosen: list
            Indices of the samples that are chosen.
        '''
        assert len(X) >= n
        if self.batch == 'auto':
            batch = self.k * n
        else:
            assert self.batch >= n, "Batch size must be greater than n!"
            batch = self.batch

        if isinstance(random_state, np.random.Generator):
            rng = random_state
        elif random_state is not None:
            rng = np.random.Generator(np.random.PCG64(random_state))
        else:
            rng = np.random.default_rng()

        return _pick(
            lambda X: self.kernel(X, **self.kernel_options),
            X,
            rng.permutation(len(X)),
            n,
            batch,
            self.k
        )


@nb.jit(
    nb.intc[:](
        nbtypes.pyobject,
        nbtypes.pyobject,
        nb.intc[:],
        nb.intc,
        nb.intc,
        nb.intc
    ),
    forceobj=True
)
def _pick(kernel, X, active, n, batch, k):
    if len(active) > batch:
        stops = np.linspace(0, len(active), k + 1, dtype=np.int)
        active = np.concatenate([
            _pick(kernel, X, active[b:e], batch // k, batch, k)
            for b, e in zip(stops[:-1], stops[1:])
        ])

    K = kernel(X[active])
    chosen = []
    for _ in range(n):
        L = np.sum(K**2, axis=1)
        L[chosen] = -np.inf  # ensure chosen points won't be selected again
        i = np.argmax(L)
        chosen.append(i)
        v = K[i, :] / np.linalg.norm(K[i, :])
        K -= np.outer(K @ v, v)

    return active[chosen]
