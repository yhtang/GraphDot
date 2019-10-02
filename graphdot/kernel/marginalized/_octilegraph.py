#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from graphdot.codegen.typetool import cpptype, rowtype
from graphdot.cuda.array import umlike, umzeros

__all__ = ['OctileGraph']

# only works with python >= 3.6
# @cpptype(upper=np.int32, left=np.int32, nzmask=np.int64, elements=np.uintp)
@cpptype([('upper', np.int32), ('left', np.int32), ('nzmask', '<u8'),
          ('p_elements', np.uintp)])
class Octile(object):
    def __init__(self, upper, left, nzmask, elements):
        self.upper = upper
        self.left = left
        self.nzmask = nzmask
        self.elements = elements

    @property
    def p_elements(self):
        return self.elements.ptr


# only works with python >= 3.6
# @cpptype(n_node=np.int32, n_octile=np.int32, degree=np.uintp,
#          node=np.uintp, octile=np.uintp)
@cpptype([('n_node', np.int32), ('n_octile', np.int32), ('p_degree', np.uintp),
          ('p_node', np.uintp), ('p_octile', np.uintp)])
class OctileGraph(object):
    """
    struct graph_t {
        int n_node, n_octile;
        deg_t    * degree;
        node_t   * node;
        octile_t * octile;
    };
    """

    def __init__(self, graph):

        nodes = graph.nodes
        edges = graph.edges
        self.n_node = len(nodes)

        ''' add phantom label if none exists to facilitate C++ interop '''
        if len(nodes.columns) == 0:
            nodes = nodes.assign(labeled=lambda _: False)

        if len(edges.columns) == 1:
            assert(edges.columns[0] == '!ij')
            edges = edges.assign(labeled=lambda _: False)

        ''' determine node type '''
        self.node_type = node_type = rowtype(nodes)
        self.node = umlike(nodes[list(node_type.names)]
                           .to_records(index=False).astype(node_type))

        ''' determine whether graph is weighted, determine edge type,
            and compute node degrees '''
        self.degree = degree = umzeros(self.padded_size, dtype=np.float32)
        edge_label_type = rowtype(edges, exclude=['!ij', '!w'])
        if '!w' in edges.columns:  # weighted graph
            self.weighted = True
            edge_type = np.dtype([('weight', np.float32),
                                  ('label', edge_label_type)], align=True)
            self.edge_type = edge_type
            for (i, j), w in zip(edges['!ij'], edges['!w']):
                degree[i] += w
                degree[j] += w
        else:
            self.weighted = False
            self.edge_type = edge_type = edge_label_type
            for i, j in edges['!ij']:
                degree[i] += 1.0
                degree[j] += 1.0

        ''' collect non-zero edge octiles '''
        uniq_oct = np.unique([(i - i % 8, j - j % 8)
                              for i, j in edges['!ij']], axis=0)
        uniq_oct = np.unique(np.vstack((uniq_oct, uniq_oct[:, -1::-1])),
                             axis=0)
        octile_dict = {(upper, left): [np.uint64(), umzeros(64, edge_type)]
                       for upper, left in uniq_oct}

        for index, row in edges.iterrows():
            i, j = row['!ij']
            if self.weighted:
                edge = (row['!w'], tuple(row[key]
                                         for key in edge_type['label'].names))
            else:
                edge = tuple(row[key] for key in edge_type.names)
            r = i % 8
            c = j % 8
            upper = i - r
            left = j - c
            octile_dict[(upper, left)][0] |= np.uint64(1 << (r * 8 + c))
            octile_dict[(upper, left)][1][r + c * 8] = edge
            octile_dict[(left, upper)][0] |= np.uint64(1 << (c * 8 + r))
            octile_dict[(left, upper)][1][c + r * 8] = edge

        ''' create edge octiles on GPU '''
        self.octile_list = [Octile(upper, left, nzmask, elements)
                            for (upper, left), (nzmask, elements)
                            in octile_dict.items()]
        self.n_octile = len(self.octile_list)

        ''' collect edge octile structures into continuous buffer '''
        self.octile_hdr = umlike(np.array([x.state for x in self.octile_list],
                                          Octile.dtype))

    @property
    def p_octile(self):
        return self.octile_hdr.ptr

    @property
    def p_degree(self):
        return self.degree.ptr

    @property
    def p_node(self):
        return self.node.ptr

    @property
    def padded_size(self):
        return (self.n_node + 7) & ~7
