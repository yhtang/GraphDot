#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import scipy.linalg


def chol_solve(A, b):
    L = np.linalg.cholesky(A)
    x = scipy.linalg.solve_triangular(
            L,
            scipy.linalg.solve_triangular(
                L, b,
                lower=True,
                check_finite=False),
            trans='C',
            lower=True,
            check_finite=False
        )
    return x


class CholSolver:
    def __init__(self, A):
        self.L = np.linalg.cholesky(A)

    def __matmul__(self, b):
        return scipy.linalg.solve_triangular(
            self.L,
            scipy.linalg.solve_triangular(
                self.L, b,
                lower=True,
                check_finite=False),
            trans='C',
            lower=True,
            check_finite=False
        )
