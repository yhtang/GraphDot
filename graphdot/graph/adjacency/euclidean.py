#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Convert spatial distances between nodes to edge adjacencies
"""
import numpy as np


class Gaussian:
    def __call__(self, d, length_scale):
        return np.exp(-0.5 * (d / length_scale)**2)


class Tent:
    def __init__(self, ord):
        self.ord = ord

    def __call__(self, d, length_scale):
        cutoff = length_scale * 3
        s = 1 - d / cutoff
        return s ** self.ord if s >= 0 else 0
