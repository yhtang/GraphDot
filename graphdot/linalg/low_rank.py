#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Low-rank approximation of square matrices.'''
import numpy as np


class FactorApprox:
    '''Represents an N-by-N square matrix A as L @ R, where L and R are N-by-k
    and k-by-N (k << N) rectangular matrices.'''

    def __init__(self, lhs, rhs):
        self._lhs = lhs
        self._rhs = rhs

    def __repr__(self):
        return f'{self.lhs.shape} @ {self.rhs.shape}'

    @property
    def lhs(self):
        return self._lhs

    @property
    def rhs(self):
        return self._rhs

    @property
    def T(self):
        return FactorApprox(self.rhs.T, self.lhs.T)

    def __neg__(self):
        return FactorApprox(-self.lhs, self.rhs)

    def __add__(self, other):
        if isinstance(other, FactorApprox):
            return BilinearSums((self, other))
        elif isinstance(other, BilinearSums):
            return BilinearSums((self, *other.factors))
        else:
            raise

    def __sub__(self, other):
        if isinstance(other, FactorApprox):
            return BilinearSums((self, -other))
        elif isinstance(other, BilinearSums):
            return BilinearSums((self, *[-f for f in other.factors]))
        else:
            raise

    def __matmul__(self, other):
        if isinstance(other, BilinearSums):
            return BilinearSums(tuple(
                [self @ b for b in other.factors]
            ))
        elif isinstance(other, FactorApprox):
            return FactorApprox(self.lhs, (self.rhs @ other.lhs) @ other.rhs)
        else:
            return self.lhs @ (self.rhs @ other)

    def todense(self):
        return self.lhs @ self.rhs

    def diagonal(self):
        return np.sum(self.lhs * self.rhs.T, axis=1)

    def trace(self):
        return self.diagonal().sum()

    def quadratic(self, a, b):
        '''Computes a @ X @ b.'''
        return (a @ self.lhs) @ (self.rhs @ b)

    def quadratic_diag(self, a, b):
        '''Computes diag(a @ X @ b).'''
        return np.sum((a @ self.lhs) * (self.rhs @ b), axis=1)


class BilinearSums(FactorApprox):
    '''Represents summations of factor approximations. Due to the bilinear
    nature of matrix inner product, it is best to store the summation as-is so
    as to preserve the low-rank structure of the matrices.'''

    def __init__(self, factors):
        self.factors = factors

    def __repr__(self):
        return ' + '.join([f'({repr(f)})' for f in self.factors])

    @property
    def T(self):
        return BilinearSums([f.T for f in self.factors])

    def __neg__(self):
        return BilinearSums([-f for f in self.factors])

    def __add__(self, other):
        if isinstance(other, FactorApprox):
            return BilinearSums((*self.factors, other))
        elif isinstance(other, BilinearSums):
            return BilinearSums((*self.factors, *other.factors))
        else:
            raise

    def __matmul__(self, other):
        if isinstance(other, BilinearSums):
            return BilinearSums(tuple(
                [a @ b for a in self.factors for b in other.factors]
            ))
        elif isinstance(other, FactorApprox):
            return BilinearSums(tuple(
                [a @ other for a in self.factors]
            ))
        else:
            raise

    def trace(self):
        return np.sum([f.trace() for f in self.factors])

    def quadratic(self, a, b):
        '''Computes a @ X @ b.'''
        return np.sum([f.quadratic(a, b) for f in self.factors], axis=0)

    def todense(self):
        return np.sum([f.todense() for f in self.factors], axis=0)


class SpectralApprox(FactorApprox):
    '''A special case of factor approximation where the matrix is symmetric and
    positive-semidefinite. In this case, the matrix can be represented using a
    spectral decomposition.'''

    def __init__(self, X, rcut=0, acut=0):
        if isinstance(X, np.ndarray):
            U, S, _ = np.linalg.svd(X, full_matrices=False)
            mask = (S >= S.max() * rcut) & (S >= acut)
            self.U = U[:, mask]
            self.S = S[mask]
        elif isinstance(X, tuple) and len(X) == 2:
            self.U, self.S = X
        self._lhs = self.U * self.S

    @property
    def lhs(self):
        return self._lhs

    @property
    def rhs(self):
        return self._lhs.T

    def diagonal(self):
        return np.sum(self.lhs**2, axis=1)

    def inverse(self):
        return SpectralApprox((self.U, 1 / self.S))

    def logdet(self):
        return 2 * np.log(self.S).sum()

    def cond(self):
        return self.S.max() / self.S.min()

    def __pow__(self, exp):
        return SpectralApprox((self.U, self.S**exp))


def dot(X, Y=None, rcut=0, acut=0):
    '''A utility method that creates factor-approximated matrix objects.'''
    if Y is None:
        return SpectralApprox(X, rcut=rcut, acut=acut)
    else:
        return FactorApprox(X, Y)
