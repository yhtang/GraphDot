#!/usr/bin/env python
# -*- coding: utf-8 -*-
import itertools
import uuid
import numpy as np
import pandas as pd
from graphdot.graph import Graph
from graphdot.util import add_classmethod
from graphdot.kernel.marginalized import MarginalizedGraphKernel
from graphdot.kernel.marginalized.basekernel import KroneckerDelta
from graphdot.kernel.marginalized.basekernel import SquareExponential
from graphdot.kernel.marginalized.basekernel import TensorProduct
from graphdot.graph.adjacency.euclidean import Tent


class SimpleTentAtomicAdjacency:
    def __init__(self, h=1.0, order=1):
        self.adj = Tent(h * 3, order)

    def __call__(self, atom1, atom2, images, cell):
        dx = atom1.position - atom2.position
        rmin = np.linalg.norm(dx)
        for ix, iy, iz in images:
            d = dx + cell[0] * ix + cell[1] * iy + cell[2] * iz
            r = np.linalg.norm(d)
            if r < rmin:
                rmin = r
        return self.adj(np.linalg.norm(rmin)), rmin

    @property
    def cutoff(self):
        return self.adj.h * 3


@add_classmethod(Graph)
def from_ase(cls, atoms, use_pbc=True, adjacency='default'):

    nodes = pd.DataFrame()
    nodes['element'] = atoms.get_atomic_numbers().astype(np.int8)

    if adjacency == 'default':
        adj = SimpleTentAtomicAdjacency(h=1.0, order=1)
    else:
        adj = adjacency

    images = list(itertools.product(*tuple([-1, 0, 1] if p else [0]
                                           for p in np.logical_and(atoms.pbc,
                                                                   use_pbc))))

    edge_data = []
    for atom1 in atoms:
        for atom2 in atoms:
            if atom2.index <= atom1.index:
                continue
            w, r = adj(atom1, atom2, images, atoms.cell)
            if w > 0:
                edge_data.append(((atom1.index, atom2.index), w, r))

    edges = pd.DataFrame(edge_data, columns=['!ij', '!w', 'length'])

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
    from ase import Atoms

    # molecules = [molecule('H2'), molecule('O2'), molecule('H2O'), molecule('CH4'), molecule('CH3OH')]
    # for m in molecules:
    #     print(m)

    m = Atoms('H3', [[0, 0, 0], [1, 0, 0], [2, 0, 0]])
    m.pbc = [True, False, False]
    m.cell = np.array([[3.0, 0, 0], [0, 0, 0], [0, 0, 0]])

    g = Graph.from_ase(m, use_pbc=True)
    print(g.edges)



    # G = [Graph.from_ase(m) for m in molecules]
    #
    # kernel = Tang2019MolecularKernel(element_prior=0.5)
    # np.set_printoptions(precision=4, suppress=True)
    # K = kernel(G)
    # D = np.diag(np.diag(K)**-0.5)
    # print(D.dot(K).dot(D))
