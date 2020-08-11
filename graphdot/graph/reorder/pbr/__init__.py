#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from .mnom import PbrMnom


_context = PbrMnom(os.path.join(os.path.dirname(__file__), 'mnom-base.ini'))


def pbr(g):
    '''Compute a partition-based permutation of a graph that minimizes the
    number of cross-tile messages. Note that the method does NOT modify the
    graph, but rather just returns a permutation vector that can be used by
    Graph.permute to achieve the actual reordering.

    Parameters
    ----------
    g: graphdot.Graph
        The graph to be reordered.

    Returns
    -------
    perm: numpy.ndarray
        Array of permuted node indices.
    '''
    A = g.adjacency_matrix.tocoo()
    perm = _context.partition(A.row, A.col, *A.shape)
    return perm
