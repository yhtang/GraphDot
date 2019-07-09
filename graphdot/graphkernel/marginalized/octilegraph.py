#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from pycuda.gpuarray import to_gpu
from graphdot.codegen.typetool import cpptype, rowtype

__all__ = ['OctileGraph']

# only works with python >= 3.6
# @cpptype(upper=np.int32, left=np.int32, nzmask=np.int64, elements=np.uintp)
@cpptype([('upper', np.int32), ('left', np.int32), ('nzmask', np.int64),
          ('elements', np.uintp)])
class Octile(object):
    """
    using octile_t = struct {
        int upper, left;
        std::int64_t nzmask;
        edge_t * elements;
    };
    """

    def __init__(self, upper, left, nzmask, elements):
        self.upper = upper
        self.left = left
        self.nzmask = nzmask
        self.__elements = to_gpu(elements)

    @property
    def elements(self):
        return self.__elements.ptr

# only works with python >= 3.6
# @cpptype(n_node=np.int32, n_octile=np.int32, degree=np.uintp,
#          node=np.uintp, octile=np.uintp)
@cpptype([('n_node', np.int32), ('n_octile', np.int32), ('degree', np.uintp),
          ('node', np.uintp), ('octile', np.uintp)])
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
        self.n_node = len(graph.nodes)

        ''' determine node type '''
        node_type = rowtype(graph.nodes)
        node_d = to_gpu(graph.nodes[list(node_type.names)]
                        .to_records(index=False)
                        .astype(node_type))

        ''' determine whether graph is weighted, determine edge type,
            and compute node degrees '''
        degree_h = np.zeros(self.padded_size, dtype=np.float32)
        edge_label_type = rowtype(graph.edges.drop(['!ij', '!w'],
                                  axis=1,
                                  errors='ignore'))
        if '!w' in graph.edges.columns:  # weighted graph
            self.weighted = True
            edge_type = np.dtype([('weight', np.float32),
                                  ('label', edge_label_type)], align=True)
            for (i, j), w in zip(graph.edges['!ij'], graph.edges['!w']):
                degree_h[i] += w
                degree_h[j] += w
        else:
            self.weighted = False
            edge_type = edge_label_type
            for i, j in graph.edges['!ij']:
                degree_h[i] += 1.0
                degree_h[j] += 1.0
        degree_d = to_gpu(degree_h)

        ''' collect non-zero edge octiles '''
        uniq_oct = np.unique([(i - i % 8, j - j % 8)
                              for i, j in graph.edges['!ij']], axis=0)
        uniq_oct = np.unique(uniq_oct + uniq_oct[:, -1::-1], axis=0)

        octile_table = {(upper, left): [0, np.zeros(64, dtype=edge_type)]
                        for upper, left in uniq_oct}

        for index, row in graph.edges.iterrows():
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
            octile_table[(upper, left)][0] |= 1 << (r + c * 8)
            octile_table[(upper, left)][1][r + c * 8] = edge
            octile_table[(left, upper)][0] |= 1 << (c + r * 8)
            octile_table[(left, upper)][1][c + r * 8] = edge

        ''' create edge octiles on GPU '''
        octile_list = [Octile(upper, left, nzmask, elements)
                       for (upper, left), (nzmask, elements)
                       in octile_table.items()]

        ''' collect edge octile structures into continuous buffer '''
        octile_hdr = to_gpu(np.array([x.state for x in octile_list],
                                     Octile.dtype))

        self.node_type = node_type
        self.edge_type = edge_type

        self.n_octile = len(octile_list)
        self.degree_d = degree_d
        self.node_d = node_d
        self.octile_hdr_d = octile_hdr
        self.octile_list = octile_list  # prevent automatic deconstruction

    @property
    def octile(self):
        return self.octile_hdr_d.ptr

    @property
    def degree(self):
        return self.degree_d.ptr

    @property
    def node(self):
        return self.node_d.ptr

    @property
    def padded_size(self):
        return (self.n_node + 7) & ~7
