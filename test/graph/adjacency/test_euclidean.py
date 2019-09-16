#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pytest
from graphdot.graph.adjacency.euclidean import Gaussian
from graphdot.graph.adjacency.euclidean import Tent

adjacencies = [
    # Gaussian(1.0),
    # Gaussian(3.0),
    # Tent(1),
    # Tent(1),
    # Tent(2),
]


# @pytest.mark.parametrize("adj", adjacencies)
# def test_euclidean_adjacency(adj):
#     assert(adj(0) == 1)
#     assert(adj(0) >= adj(1))
#     assert(adj(np.inf) == 0)
#     np.random.seed(0)
#     for x, y in np.random.lognormal(0.0, 3.0, (1000, 2)):
#         assert(adj(x) >= adj(x + y))
