import sys
from scipy.io import mmread
from scipy.io import mmwrite
from scipy.sparse import coo_matrix
import numpy as np
import Util
from scipy.sparse.csgraph import reverse_cuthill_mckee

import pbr

A = Util.readtxtmat(sys.argv[1])
assert A.shape[0] == A.shape[1]

res = pbr.run(A.row, A.col, A.shape[0], A.shape[1], len(A.data), 8, 100,
              "mnom-base.ini")

perm = [(v, res[v]) for v in range(A.shape[0])]
perm = sorted(perm, key=lambda x:x[1])
perm = [x[0] for x in perm]

tot, emp, occ, fillr = Util.tile_stats(A, 8)
sys.stdout.write('dims %4d %4d %4d --- orig tot %4d emp %4d occ %4d '
                 'fillr %3.1f occ-tile-ratio %3.1f' %
                 (A.shape[0], A.shape[1], A.nnz,
                  tot, emp, occ, fillr * 100,
                  float(occ)/float(tot) * 100))
sys.stdout.write('\n')

Aperm = Util.perm_sym(A, perm)
tot, emp, occ, fillr = Util.tile_stats(Aperm, 8)
sys.stdout.write('PBR tot %4d emp %4d occ %4d fillr %3.1f '
                     'occ-tile-ratio %3.1f' %
                     (tot, emp, occ, fillr * 100,
                      float(occ)/float(tot) * 100))
sys.stdout.write('\n')


perm = reverse_cuthill_mckee(A.tocsr(), symmetric_mode=True)
Aperm = Util.perm_sym(A, perm)
tot, emp, occ, fillr = Util.tile_stats(Aperm, 8)
sys.stdout.write('RCM tot %4d emp %4d occ %4d fillr %3.1f '
                     'occ-tile-ratio %3.1f' %
                     (tot, emp, occ, fillr * 100,
                      float(occ)/float(tot) * 100))
sys.stdout.write('\n')

# print(perm)
