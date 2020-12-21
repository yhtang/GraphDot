#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import scipy.sparse.linalg


class CGSolver:
    def __init__(self, A, **options):
        self.A = A
        self.options = dict(**options)

    def __matmul__(self, b):
        x, info = scipy.sparse.linalg.cg(self.A, b, **self.options)
        if info == 0:
            return x
        else:
            raise RuntimeError(
                f'CG solver failed with error code {info}.'
            )

    def todense(self):
        return self @ np.eye(*self.A.shape)

    def diagonal(self):
        return self.todense().diagonal()
