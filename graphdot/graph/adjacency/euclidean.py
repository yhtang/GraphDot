#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Convert spatial distances between nodes to edge adjacencies
"""
import numpy as np
# import numba as nb


# @nb.jitclass([('half_inv_sigma2', nb.float32)])
class Gaussian:
    def __init__(self, h):
        self.half_inv_h2 = 0.5 / h**2

    def compute(self, d):
        return np.exp(d**2 * self.half_inv_h2)


class Tent:
    def __init__(self, ord):
        self.ord = ord

    def __call__(self, d, length_scale):
        s = 1 - d / self.cutoff(length_scale)
        return s ** self.ord if s >= 0 else 0

    def cutoff(self, length_scale):
        return length_scale * 3


if __name__ == '__main__':

    g = Gaussian(1.0)
    print(g(1.0))
    print(g(2.0))
    print(g(3.0))
