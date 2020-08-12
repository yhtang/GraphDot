#!/usr/bin/env python
# -*- coding: utf-8 -*-
from .mnom import PbrMnom


_partitioner = PbrMnom()


def pbr(g, partitioner=None):
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
    if partitioner is None:
        partitioner = _partitioner
    A = g.adjacency_matrix.tocoo()
    perm = partitioner(A.row, A.col, *A.shape)
    return perm
