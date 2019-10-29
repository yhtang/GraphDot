#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from graphdot.codegen.typetool import cpptype, rowtype
from graphdot.cuda.array import umlike, umzeros, umempty

__all__ = ['OctileGraph']

# only works with python >= 3.6
# @cpptype(n_node=np.int32, n_octile=np.int32, degree=np.uintp,
#          node=np.uintp, octile=np.uintp)
@cpptype([('n_node', np.int32), ('n_octile', np.int32), ('p_degree', np.uintp),
          ('p_node', np.uintp), ('p_octile', np.uintp)])
class OctileGraph(object):
    """ Python counterpart of C++ class graphdot::graph_t """

    # @cpptype(upper=np.int32, left=np.int32, nzmask=np.int64, elements=np.uintp)
    @cpptype([('elements', np.uintp), ('nzmask', '<u8'), ('nzmask_r', '<u8'),
              ('upper', np.int32), ('left', np.int32)])
    class Octile(object):
        def __init__(self, upper, left, nzmask, nzmask_r, elements):
            self.upper = upper
            self.left = left
            self.nzmask = nzmask
            self.nzmask_r = nzmask_r
            self.elements = elements

    def __init__(self, graph):

        nodes = graph.nodes
        edges = graph.edges
        self.n_node = len(nodes)

        ''' add phantom label if none exists to facilitate C++ interop '''
        if len(nodes.columns) == 0:
            nodes = nodes.assign(labeled=lambda _: False)

        assert(len(edges.columns) >= 2)
        if len(edges.columns) == 2:
            assert('!i' in edges.columns and '!j' in edges.columns)
            edges = edges.assign(labeled=lambda _: False)

        ''' determine node type '''
        self.node_type = node_type = rowtype(nodes)
        self.node = umlike(nodes[list(node_type.names)]
                           .to_records(index=False).astype(node_type))

        ''' determine whether graph is weighted, determine edge type,
            and compute node degrees '''
        self.degree = degree = umzeros(self.n_node, dtype=np.float32)
        edge_label_type = rowtype(edges, exclude=['!i', '!j', '!w'])
        if '!w' in edges.columns:  # weighted graph
            self.weighted = True
            edge_type = np.dtype([('weight', np.float32),
                                  ('label', edge_label_type)], align=True)
            self.edge_type = edge_type
            np.add.at(degree, edges['!i'], edges['!w'])
            np.add.at(degree, edges['!j'], edges['!w'])

            # print('edge_label_type', edge_label_type)
            if edge_label_type.itemsize != 0:
                edge_aos = np.fromiter(zip(edges['!w'],
                                        zip(*[edges[t] for t in edge_label_type.names])
                                        ), dtype=edge_type)
            else:
                edge_aos = np.fromiter(zip(edges['!w'], [None] * len(edges)), dtype=edge_type)
            # print('edge_aos\n', edge_aos, sep='')

        else:
            self.weighted = False
            self.edge_type = edge_type = edge_label_type
            np.add.at(degree, edges['!i'], 1.0)
            np.add.at(degree, edges['!j'], 1.0)
            edge_aos = np.fromiter(zip(*[edges[t] for t in edge_type.names]),
                                   dtype=edge_type)

        ''' collect non-zero edge octiles '''
        # uniq_oct = np.unique([(i - i % 8, j - j % 8)
        #                       for i, j in zip(edges['!i'], edges['!j'])], axis=0)

        nnz = len(edges)
        # print('edges\n', edges, sep='')

        indices = np.empty((4, nnz * 2), dtype=np.uint32, order='C')
        i, j, up, lf = indices
        i[:nnz] = j[nnz:] = edges['!i']
        j[:nnz] = i[nnz:] = edges['!j']
        up[:] = i - i % 8
        lf[:] = j - j % 8
        order = np.lexsort(indices, axis=0)
        perm = order % nnz
        indices[:, :] = indices[:, order]
        # print(order)
        # print(perm)
        # print('indices.T\n', indices.T, sep='')

        diff = np.empty(nnz * 2)
        diff[0] = True
        diff[1:] = (up[:-1] != up[1:]) | (lf[:-1] != lf[1:])

        octile_starts = np.flatnonzero(diff)

        # print(octile_starts)

        # compose nzmasks

        np.set_printoptions(linewidth=9999)

        nzmasks = np.bitwise_or.reduceat(1 << (i - up + (j - lf) * 8).astype(np.uint64), octile_starts)
        nzmasks_r = np.bitwise_or.reduceat(1 << (j - lf + (i - up) * 8).astype(np.uint64), octile_starts)

        # print('i - up\n', i - up, sep='')
        # print('j - lf\n', j - lf, sep='')

        # print('i - up + (j - lf) * 8\n', i - up + (j - lf) * 8, sep='')
        # print('1 << (i - up + (j - lf) * 8)\n', 1 << (i - up + (j - lf) * 8), sep='')
        # print('i j mask\n', np.column_stack((i, j, 1 << np.array(i - up + (j - lf) * 8, dtype=np.uint64))), sep='')

        # print('nzmasks')
        # import sys
        # for m in nzmasks:
        #     print('%016X' % m)
        #     for r in range(8):
        #         for c in range(8):
        #             sys.stdout.write('# ' if m & np.uint64(1 << (r + c * 8)) else '. ')
        #         sys.stdout.write('\n')

        self.n_octile = n_octile = len(octile_starts)
        self.elements = elements = umempty(nnz * 2, edge_type)
        self.octiles = octiles = umempty(n_octile, self.Octile.dtype)

        # print(type(elements.base))
        # print(int(elements.base))

        elements[:] = edge_aos[perm]

        # print('elements\n', elements, sep='')

        # print('edge_type.itemsize', edge_type.itemsize)

        octiles[:] = list(zip(int(elements.base) + octile_starts * edge_type.itemsize,
                              nzmasks,
                              nzmasks_r,
                              up[octile_starts],
                              lf[octile_starts])
                              )
        
        # print('octiles')
        # for octile in octiles:
        #     print('OCTILE %4d %4d %016X %016X %24d' % tuple(octile))

        # # octile_dict = {(upper, left): [np.uint64(), np.uint64(),
        # #                                umzeros(64, edge_type)]
        # #                for upper, left in uniq_oct}

        # # for index, row in edges.iterrows():
        # #     i, j = int(row['!i']), int(row['!j'])
        # #     if self.weighted:
        # #         edge = (row['!w'], tuple(row[key]
        # #                                  for key in edge_type['label'].names))
        # #     else:
        # #         edge = tuple(row[key] for key in edge_type.names)
        # #     r = i % 8
        # #     c = j % 8
        # #     upper = i - r
        # #     left = j - c
        # #     octile_dict[(upper, left)][0] |= np.uint64(1 << (r + c * 8))
        # #     octile_dict[(upper, left)][1] |= np.uint64(1 << (c + r * 8))
        # #     octile_dict[(upper, left)][2][r + c * 8] = edge
        # #     octile_dict[(left, upper)][0] |= np.uint64(1 << (c + r * 8))
        # #     octile_dict[(left, upper)][1] |= np.uint64(1 << (r + c * 8))
        # #     octile_dict[(left, upper)][2][c + r * 8] = edge

        # ''' create edge octiles on GPU '''
        # self.octile_list = [Octile(upper, left, nzmask, nzmask_r, elements)
        #                     for (upper, left), (nzmask, nzmask_r, elements)
        #                     in octile_dict.items()]
        # # compact the tiles
        # for o in self.octile_list:
        #     k = 0
        #     for i in range(64):
        #         if o.nzmask & np.uint64(1 << i):
        #             o.elements[k] = o.elements[i]
        #             k += 1

        # self.n_octile = len(self.octile_list)

        # ''' collect edge octile structures into continuous buffer '''
        # self.octile_hdr = umlike(np.array([x.state for x in self.octile_list],
        #                                   Octile.dtype))

    @property
    def p_octile(self):
        # print('int(self.octiles.base) %x' % int(self.octiles.base))
        return int(self.octiles.base)

    @property
    def p_degree(self):
        # print('int(self.degree.base) %x' % int(self.degree.base))
        return int(self.degree.base)

    @property
    def p_node(self):
        # print('int(self.node.base) %x' % int(self.node.base))
        return int(self.node.base)
