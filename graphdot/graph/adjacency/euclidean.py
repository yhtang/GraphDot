#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Convert spatial distances between nodes to edge adjacencies
"""
import numpy as np


class Gaussian:
    def __init__(self, length_scale):
        self.length_scale = length_scale

    def __call__(self, d):
        return np.exp(-0.5 * (d / self.length_scale)**2)


class Tent:
    def __init__(self, cut, ord):
        self.cut = cut
        self.ord = ord

    def __call__(self, d):
        s = 1 - d / self.cut
        return s**self.ord if s >= 0 else 0
