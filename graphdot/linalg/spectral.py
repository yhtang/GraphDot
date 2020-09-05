#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np


def powerh(H, p, rcond=None, return_symmetric=True, return_eigvals=False):
    r'''Compute the fractional power of a Hermitian matrix as defined through
    eigendecomposition.

    Parameters
    ----------
    H: matrix
        H must be symmetric/self-conjugate, a.k.a. Hermitian.
    p: float
        The power to be raised.
    rcond: float
        Cutoff for small eigenvalues. Eigenvalues less than or equal to
        `rcond * largest_eigenvalue` are discarded.
    return_symmetric: bool
        Whether or not to make the returned matrix symmetric by multiplying it
        with the transposed eigenvectors on the right hand side.

    Returns
    -------
    L: matrix
        :py:math:`H^p`.
    '''
    a, Q = np.linalg.eigh(H)
    if rcond is not None:
        mask = a > a.max() * rcond
        a = a[mask]
        Q = Q[:, mask]
    if np.any(a <= 0) and p < 1 and p != 0:
        raise np.linalg.LinAlgError(
            f'Cannot raise a non-positive definite matrix to a power of {p}.'
        )
    Hp = Q * a**p
    if return_symmetric:
        Hp = Hp @ Q.T

    return (Hp, a) if return_eigvals is True else Hp


def pinvh(H, rcond=1e-10, estimate_logdet=False):
    r'''Compute the pseudoinverse of a Hermitian matrix.

    Parameters
    ----------
    H: matrix
        H must be symmetric/self-conjugate, a.k.a. Hermitian.
    rcond: float
        Cutoff for small eigenvalues. Eigenvalues less than or equal to
        `rcond * largest_eigenvalue` and associated eigenvators are discarded
        in forming the pseudoinverse.
    estimate_logdet: bool
        Whether or not to compute an estimate of the log determinant of the
        matrix while correcting small negative eigenvalues.

    Returns
    -------
    H_inv: matrix
        :py:math:`H^{-1}`.
    logdet: float, optional if estimate_logdet is True
        An estimate of the log-determinant of H.
    '''
    a, Q = np.linalg.eigh(H)
    beta = a.max() * rcond
    mask = a > beta
    H_inv = (Q[:, mask] / a[mask]) @ Q.T[mask, :]
    if estimate_logdet is True:
        logdet = np.sum(np.log(np.maximum(a, beta)))
        return H_inv, logdet
    else:
        return H_inv
