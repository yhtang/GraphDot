#!/usr/bin/env python
# -*- coding: utf-8 -*-
import uuid
import numpy as np
import pandas as pd
from graphdot import Graph
from graphdot.graphkernel.marginalized import MarginalizedGraphKernel
from graphdot.graphkernel.marginalized.basekernel import KroneckerDelta
from graphdot.graphkernel.marginalized.basekernel import SquareExponential
from graphdot.graphkernel.marginalized.basekernel import TensorProduct


def add_classmethod(cls):
    def decorate(func):
        if hasattr(cls, func.__name__):
            raise RuntimeWarning(' '.join(['Overriding', repr(cls),
                                           'existing method', repr(func)]))
        setattr(cls, func.__name__, classmethod(func))
    return decorate


@add_classmethod(Graph)
def from_ase(cls, atoms, adjacency='tent', bonding_distance='default',
             bonding_zoom=5.0):

    from graphdot.graph.adjacency.euclidean import Tent

    nodes = pd.DataFrame()
    nodes['element'] = atoms.get_atomic_numbers().astype(np.int8)

    n = len(atoms)
    adj = Tent(bonding_zoom, 2)
    x = atoms.get_positions()
    ij, weight, length = [], [], []
    for i in range(n):
        for j in range(i+1, n):
            w = adj(x[i], x[j])
            if w > 0:
                ij.append((i, j))
                weight.append(w)
                length.append(np.linalg.norm(x[i] - x[j]))

    edges = pd.DataFrame()
    edges['!ij'] = ij
    edges['!w'] = np.array(weight).astype(np.float32)
    edges['length'] = length

    return cls(nodes, edges, title='ASE Atoms {formula} {id}'.format(
               formula=atoms.get_chemical_formula(), id=uuid.uuid4().hex))


class Tang2019MolecularKernel(object):

    def __init__(self, **kwargs):
        self.stopping_probability = kwargs.pop('stopping_probability', 0.01)
        self.element_prior = kwargs.pop('element_prior', 0.2)
        self.edge_length_scale = kwargs.pop('edge_length_scale', 0.05)
        self._makekernel()

    def _makekernel(self):
        self.kernel = MarginalizedGraphKernel(
            TensorProduct(element=KroneckerDelta(self.element_prior, 1.0)),
            TensorProduct(length=SquareExponential(self.edge_length_scale)),
            q=self.stopping_probability
        )

    def __call__(self, X, Y=None):
        return self.kernel(X, Y)


if __name__ == '__main__':

    from ase.build import molecule

    molecules = [molecule('H2'), molecule('O2'), molecule('H2O'), molecule('CH4'), molecule('CH3OH')]
    for m in molecules:
        print(m)

    G = [Graph.from_ase(m) for m in molecules]

    kernel = Tang2019MolecularKernel(element_prior=0.5)
    np.set_printoptions(precision=4, suppress=True)
    K = kernel(G)
    D = np.diag(np.diag(K)**-0.5)
    print(D.dot(K).dot(D))
