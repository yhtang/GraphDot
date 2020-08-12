#!/usr/bin/env python
# -*- coding: utf-8 -*-
import scipy.sparse


def rcm(g):
    '''Compute the reverse Cuthill-Mckee permutation of a graph. Note that the
    method does NOT modify the graph, but rather just returns a permutation
    vector that can be used by Graph.permute to achieve the actual reordering.

    Parameters
    ----------
    g: Graph
        The graph to be reordered.

    Returns
    -------
    perm: numpy.ndarray
        Array of permuted node indices.
    '''
    return scipy.sparse.csgraph.reverse_cuthill_mckee(
        g.adjacency_matrix, symmetric_mode=True
    )
