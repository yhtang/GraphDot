#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np


def binvh1(A_inv, v, d):
    r'''Computes the inversion of the block matrix ``B = [[A, v], [v.T, d]]``,
    with :py:math:`v` being a vector and :py:math:`d` being a scalar, using
    :py:math:`\mathbf{D}^{-1}` and a fast rank-one update.'''
    v = np.ascontiguousarray(v)

    w = A_inv @ v
    schur = d - v @ w

    B_inv = np.empty((A_inv.shape[0] + 1, A_inv.shape[1] + 1))
    B_inv[:-1, :][:, :-1] = A_inv + np.outer(w, w) / schur
    B_inv[-1, :-1] = B_inv[:-1, -1] = -w / schur
    B_inv[-1, -1] = 1 / schur

    return B_inv
