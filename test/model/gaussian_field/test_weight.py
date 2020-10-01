#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from scipy.spatial import distance_matrix as pairwise_distances
from graphdot.model.gaussian_field import RBFOverHausdorff


def test_weight():

    class Test_Metric():

        def __call__(self, X, Y=None):
            if Y is None:
                return pairwise_distances(X, X)
            else:
                return pairwise_distances(X, Y)
    np.random.seed(0)
    X = [(0, 1), (1, 2)]
    weight = RBFOverHausdorff(np.array([0]), np.array([[0, 1]]), X,
                              Test_Metric())
    assert np.allclose(weight(X), np.array([[1, 0.1353353], [0.1353353, 1]]))
