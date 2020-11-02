#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Low-rank approximation of square matrices.'''
import numpy as np
import scipy.sparse.linalg as splin


class LowRankBase:
    def __add__(self, other):
        return add(self, other)

    def __sub__(self, other):
        return sub(self, other)

    def __matmul__(self, other):
        return matmul(self, other)


class Sum(LowRankBase):
    '''Represents summations of factor approximations. Due to the bilinear
    nature of matrix inner product, it is best to store the summation as-is so
    as to preserve the low-rank structure of the matrices.'''

    def __init__(self, factors):
        self.factors = factors

    def __repr__(self):
        return ' + '.join([f'({repr(f)})' for f in self.factors])

    @property
    def T(self):
        return Sum([f.T for f in self.factors])

    def __neg__(self):
        return Sum([-f for f in self.factors])

    def diagonal(self):
        return np.sum([f.diagonal() for f in self.factors], axis=0)

    def trace(self):
        return np.sum([f.diagonal().sum() for f in self.factors])

    def quadratic(self, a, b):
        '''Computes a @ X @ b.'''
        return np.sum([f.quadratic(a, b) for f in self.factors], axis=0)

    def todense(self):
        return np.sum([f.todense() for f in self.factors], axis=0)


class LATR(LowRankBase):
    r'''Represents an N-by-N square matrix A as :py:math:`L \cdot R`, where L
    and R are N-by-k and k-by-N (:py:math:`k << N`) rectangular matrices.'''

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
        return LATR(self.rhs.T, self.lhs.T)

    def __neg__(self):
        return LATR(-self.lhs, self.rhs)

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
        return LATR(a @ self.lhs, self.rhs @ b).diagonal()


class LLT(LATR):
    r'''A special case of factor approximation where the matrix is symmetric
    and positive-semidefinite. In this case, the matrix can be represented as
    :py:math:`L \cdot L^\mathsf{T}` from a spectral decomposition.'''

    def __init__(self, X, rcond=0, mode='truncate'):
        if isinstance(X, np.ndarray):
            U, S, _ = np.linalg.svd(X, full_matrices=False)
            beta = S.max() * rcond
            if mode == 'truncate':
                mask = S >= beta
                self.U = U[:, mask]
                self.S = S[mask]
            elif mode == 'clamp':
                self.U = U
                self.S = np.maximum(S, beta)
            else:
                raise RuntimeError(
                    f"Unknown spectral approximation mode '{mode}'."
                )
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

    def pinv(self):
        return LLT((self.U, 1 / self.S))

    def logdet(self):
        return 2 * np.log(self.S).sum()

    def cond(self):
        return (self.S.max() / self.S.min())**2

    def __pow__(self, exp):
        return LLT((self.U, self.S**exp))


def dot(X, Y=None, method='auto', rcond=0, mode='truncate'):
    r'''A utility method that creates low-rank matrices
    :py:math:`A \doteq X \cdot Y`.

    Parameters
    ----------
    X: ndarray
        Left hand side of the product.
    Y: ndarray
        Right hand side of the product. If None, Y will be assumed to be the
        transposition of X.
    method: 'auto' or 'direct' or 'spectral'
        If 'direct', store the matrix as the product of X and Y. If 'spectral'
        and Y is None, store the matrix as the product of the singular vectors
        and singular values of X. 'auto' is equivalent to 'spectral' when Y is
        None and 'direct' otherwise.
    rcond: float
        Threshold for small singular values when method is 'spectral'.
    mode : 'truncate' or 'clamp'
        Determines how small singular values of the original matrix are
        handled. For 'truncate', small values are discarded; for 'clamp', they
        are fixed to be the product of the largest singular value and rcond.
    '''
    assert method in ['auto', 'direct', 'spectral'], f'Unknown method {method}'
    if Y is None:
        if method == 'spectral' or method == 'auto':
            return LLT(X, rcond=rcond, mode=mode)
        else:
            return LATR(X, X.T)
    else:
        if method == 'spectral':
            raise RuntimeError(
                'Spectral approximation only usable when Y is None.'
            )
        else:
            return LATR(X, Y)


def add(A, B):
    factors = A.factors if isinstance(A, Sum) else [A]
    factors += B.factors if isinstance(B, Sum) else [B]
    return Sum(factors)


def sub(A, B):
    factors = A.factors if isinstance(A, Sum) else [A]
    factors += [-f for f in B.factors] if isinstance(B, Sum) else [-B]
    return Sum(factors)


def matmul(A, B):
    if isinstance(A, Sum):
        if isinstance(B, Sum):
            return Sum([
                a @ b for a in A.factors for b in B.factors
            ])
        else:
            return Sum([
                a @ B for a in A.factors
            ])
    else:
        if isinstance(B, Sum):
            return Sum([
                A @ b for b in B.factors
            ])
        elif isinstance(B, LATR):
            return LATR(A.lhs, (A.rhs @ B.lhs) @ B.rhs)
        else:
            return A.lhs @ (A.rhs @ B)


def pinvh(A: LATR, d, k='auto', rcond=1e-10, mode='truncate'):
    '''Calculate the low-rank approximated pseudoinverse of a low-rank
    symmetric matrix with optional diagonal regularization.

    Parameters
    ----------
    A: :py:class:`LATR`.
        A low-rank symmetric positive semidefinite matrix.
    d: array
        An optional regularization vector that will be added elementwise to the
        diagonal of ``A``.
    k: int or 'auto'
        Number of eigenvalues to resolve. If 'auto', k will be set to be the
        sum of the rank of ``A`` plus the number of nonzeros in ``d``.
    rcond: float
        Cutoff for small eigenvalues. Eigenvalues less than or equal to
        `rcond * largest_eigenvalue` and associated eigenvators are discarded
        in forming the pseudoinverse.
    mode: str
        Determines how small eigenvalues of the original matrix are handled.
        For 'truncate', small eigenvalues are discarded; for 'clamp', they are
        fixed to be the product of the largest eigenvalue and rcond.

    Returns
    -------
    Ainv: :py:class:`LLT`.
        A low-rank representation of the pseudoinverse of ``A``.
    '''

    class MatVecOperator(splin.LinearOperator):

        def __init__(self, A, d):
            self.A = A
            self.d = d

        @property
        def shape(self):
            return (len(self.d), len(self.d))

        @property
        def dtype(self):
            return self.d.dtype

        def _matvec(self, b):
            return self.A @ b + self.d * b

        def _matmat(self, b):
            return self.A @ b + self.d[:, None] * b

        def _adjoint(self):
            return self

    if k == 'auto':
        k = A.lhs.shape[1] + np.count_nonzero(d)
    else:
        assert isinstance(k, int)

    a, Q = splin.eigsh(MatVecOperator(A, d), k=k)
    beta = a.max() * rcond
    mask = a > beta

    if mode == 'truncate':
        a = a[mask]
        Q = Q[:, mask]
    elif mode == 'clamp':
        a[~mask] = beta
    else:
        raise RuntimeError(f"Unknown pseudoinverse mode '{mode}'.")

    return LLT((Q, a**-0.5))
