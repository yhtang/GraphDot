#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np


# TODO: name: k-way stochastic mini-batch greedy volumn maximizer?
class GreedyVolumeMaximizer:

    def __init__(self, kernel, k=10, batch=100):
        self.kernel = kernel
        self.k = k
        self.batch = batch

    def __call__(self, X, n, random_state=None):
        '''
        '''
        assert len(X) >= n

        if isinstance(random_state, np.random.Generator):
            rng = random_state
        elif random_state is not None:
            rng = np.random.Generator(np.random.PCG64(random_state))
        else:
            rng = np.random.default_rng()

        return self._pick(X, rng.permutation(len(X)), n)

    def _pick(self, X, active, n):
        if len(active) > self.batch:
            stops = np.linspace(0, len(active), self.k + 1, dtype=np.int)
            print(f'stops\n{stops}')
            active = np.concatenate([
                self._pick(X, active[b:e], self.batch // self.k)
                for b, e in zip(stops[:-1], stops[1:])
            ])

        K = self.kernel(X[active])
        print(f'K\n{K}')
        chosen = []
        for _ in range(n):
            print(f'Norms+: {np.sum(K**2, axis=1)}')
            i = np.argmax(np.sum(K**2, axis=1))
            chosen.append(active[i])
            v = K[i, :] / np.linalg.norm(K[i, :])
            K -= np.outer(K @ v, v)
            print(f'Norms-: {np.sum(K**2, axis=1)}')

        return chosen
