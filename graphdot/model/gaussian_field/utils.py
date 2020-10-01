#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np


def dist_Hausdorff(ds, kernel, xi=1):
    """
    Computes the Hausdorff Distance Matrix for a given dataset.
    ds: Pandas DataFrame containing fields "nx" and "graph"
    kernel: Kernel Function
    xi: Multiplicative Parameter
    """
    sizes = ds.graph.apply(lambda g: len(g.nodes)).to_numpy()
    starts = np.zeros(len(ds) + 1, dtype=np.int)
    np.cumsum(sizes, out=starts[1:])

    R = kernel(ds.graph, nodal=True, lmin=0, timing=True)
    d = R.diagonal()**-0.5
    K = d[:, None] * R * d[None, :]
    K = K**xi
    D = np.sqrt(np.maximum(0, 2 - 2 * K))

    Dhc = np.minimum.reduceat(D, starts[:-1], axis=1)
    Dhr = np.maximum.reduceat(Dhc, starts[:-1], axis=0)
    return np.maximum(Dhr, Dhr.T)


def pairwise_Hausdorff(ds1, ds2, kernel):
    """
    Computes the Hausdorff Distance Matrix for a given dataset.
    ds1: Pandas DataFrame containing fields "nx" and "graph"
    ds2: Pandas DataFrame containing fields "nx" and "graph"
    kernel: Kernel Function
    """
    sizes1 = ds1.graph.apply(lambda g: len(g.nodes)).to_numpy()
    starts1 = np.zeros(len(ds1) + 1, dtype=np.int)
    np.cumsum(sizes1, out=starts1[1:])

    sizes2 = ds2.graph.apply(lambda g: len(g.nodes)).to_numpy()
    starts2 = np.zeros(len(ds2) + 1, dtype=np.int)
    np.cumsum(sizes2, out=starts2[1:])

    R12 = kernel(ds1.graph, ds2.graph, nodal=True)
    d1 = kernel.diag(ds1.graph, nodal=True)**-0.5
    d2 = kernel.diag(ds2.graph, nodal=True)**-0.5
    K12 = d1[:, None] * R12 * d2[None, :]

    D = np.sqrt(np.maximum(0, 2 - 2 * K12))

    Dhc1 = np.minimum.reduceat(D, starts2[:-1], axis=1)
    Dhr1 = np.maximum.reduceat(Dhc1, starts1[:-1], axis=0)

    Dhr2 = np.minimum.reduceat(D, starts1[:-1], axis=0)
    Dhc2 = np.maximum.reduceat(Dhr2, starts2[:-1], axis=1)
    return np.maximum(Dhr1, Dhc2)


def weight_Hausdorff(ds1, ds2, kernel):
    """
    Computes the Hausdorff Distance Matrix for a given dataset.
    ds1: Pandas DataFrame containing fields "nx" and "graph"
    ds2: Pandas DataFrame containing fields "nx" and "graph"
    kernel: Kernel Function
    """
    sizes1 = ds1.graph.apply(lambda g: len(g.nodes)).to_numpy()
    starts1 = np.zeros(len(ds1) + 1, dtype=np.int)
    np.cumsum(sizes1, out=starts1[1:])

    sizes2 = ds2.graph.apply(lambda g: len(g.nodes)).to_numpy()
    starts2 = np.zeros(len(ds2) + 1, dtype=np.int)
    np.cumsum(sizes2, out=starts2[1:])

    R12 = kernel(ds1.graph, ds2.graph, nodal=True)
    d1 = kernel.diag(ds1.graph, nodal=True)**-0.5
    d2 = kernel.diag(ds2.graph, nodal=True)**-0.5
    K12 = d1[:, None] * R12 * d2[None, :]

    Dhc1 = np.maximum.reduceat(K12, starts2[:-1], axis=1)
    Dhr1 = np.minimum.reduceat(Dhc1, starts1[:-1], axis=0)

    Dhr2 = np.maximum.reduceat(K12, starts1[:-1], axis=0)
    Dhc2 = np.minimum.reduceat(Dhr2, starts2[:-1], axis=1)
    return np.minimum(Dhr1, Dhc2)
