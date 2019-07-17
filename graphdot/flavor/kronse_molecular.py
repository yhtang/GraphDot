#!/usr/bin/env python
# -*- coding: utf-8 -*-
import itertools
import uuid
import numpy as np
import pandas as pd
from graphdot.graph import Graph
from graphdot.util import add_classmethod





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
