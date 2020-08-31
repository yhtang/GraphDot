#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Convert spatial distances between nodes to edge adjacencies
"""
import numpy as np


class Gaussian:
    def __call__(self, d, length_scale):
        return np.exp(-0.5 * d**2 / length_scale**2)

    def cutoff(self, length_scale):
        return np.inf


class Tent:
    def __init__(self, ord):
        self.ord = ord

    def __call__(self, d, length_scale):
        s = 1 - d / self.cutoff(length_scale)
        return s ** self.ord if s >= 0 else 0

    def cutoff(self, length_scale):
        return length_scale * 3
