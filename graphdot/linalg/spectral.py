#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np


def powerh(H, p, symmetric=True, return_eigvals=False):
    r'''Compute the fractional power of a Hermitian matrix as defined through
    eigendecomposition.

    Parameters
    ----------
    H: matrix
        H must be symmetric/self-conjugate, a.k.a. Hermitian.
    p: float
        The power to be raised.
    symmetri: bool
        Whether or not to multiply with the transposed eigenvectors on the
        right hand side to make the returned matrix symmetric.

    Returns
    -------
    L: matrix
        :py:math:`H^p`.
    '''
    a, Q = np.linalg.eigh(H)
    if np.any(a <= 0) and p < 0:
        raise np.linalg.LinAlgError(
            'Cannot raise a non-positive definite matrix to a negative power.'
        )
    Hp = Q * a**p
    if symmetric:
        Hp @= Q.T

    return (Hp, a) if return_eigvals is True else Hp
