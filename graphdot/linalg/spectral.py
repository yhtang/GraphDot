#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np


def powerh(H, p, rcond=None, mode='truncate', return_symmetric=True,
           return_eigvals=False):
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
    mode: str
        Determines how small eigenvalues of the original matrix are handled.
        For 'truncate', small eigenvalues are discarded; for 'clamp', they are
        fixed to be the product of the largest eigenvalue and rcond.
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
        beta = a.max() * rcond
        if mode == 'truncate':
            mask = a > beta
            a = a[mask]
            Q = Q[:, mask]
        elif mode == 'clamp':
            a = np.maximum(a, beta)
        else:
            raise RuntimeError(f"Unknown pseudoinverse mode '{mode}'.")
    if np.any(a <= 0) and p < 1 and p != 0:
        raise np.linalg.LinAlgError(
            f'Cannot raise a non-positive definite matrix to a power of {p}.'
        )
    Hp = Q * a**p
    if return_symmetric:
        Hp = Hp @ Q.T

    return (Hp, a) if return_eigvals is True else Hp


def pinvh(H, rcond=1e-10, mode='truncate', return_nlogdet=False):
    r'''Compute the pseudoinverse of a Hermitian matrix using its eigenvalue
    decomposition. Only eigenvalues larger than a certain threshold will be
    included to construct the pseudoinverse. This method differs from
    :py:method:`np.linalg.pinv` in that it uses *eigendecomposition* instead
    singular decomposition. It also differs from
    :py:method:`scipy.linalg.pinvh` in that it includes only *positive*
    eigenvalues. This design choice was made to prevent some nearly singular
    matrices, that contains elementwise error of relative magnitude 1e-7, to
    give rise to large negative log-likelihood values in Gaussian
    process regression.

    Parameters
    ----------
    H: matrix
        H must be symmetric/self-conjugate, a.k.a. Hermitian.
    rcond: float
        Cutoff for small eigenvalues. Eigenvalues less than or equal to
        `rcond * largest_eigenvalue` and associated eigenvators are discarded
        in forming the pseudoinverse.
    mode: str
        Determines how small eigenvalues of the original matrix are handled.
        For 'truncate', small eigenvalues are discarded; for 'clamp', they are
        fixed to be the product of the largest eigenvalue and rcond.
    return_nlogdet: bool
        Whether or not to return the negative log determinant of the
        pseudoinverse.

    Returns
    -------
    H_inv: matrix
        :py:math:`H^{\dagger}`.
    nlogdet: float, optional if estimate_logdet is True
        Negative log-determinant of :py:math:`H^{\dagger}` while ignoring zero
        eigenvalues.
    '''
    a, Q = np.linalg.eigh(H)
    beta = a.max() * rcond
    mask = a > beta
    if mode == 'truncate':
        a = a[mask]
        Q = Q[:, mask]
    elif mode == 'clamp':
        a[~mask] = beta
    else:
        raise RuntimeError(f"Unknown pseudoinverse mode '{mode}'.")
    H_inv = (Q / a) @ Q.T
    if return_nlogdet is True:
        nlogdet = np.sum(np.log(a))
        return H_inv, nlogdet
    else:
        return H_inv
