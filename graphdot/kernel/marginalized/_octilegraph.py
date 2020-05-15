#!/usr/bin/env python
# -*- coding: utf-8 -*-
import ctypes
import itertools as it
import json
import numpy as np
from graphdot.codegen.typetool import cpptype, common_min_type
from graphdot.cuda.array import umzeros, umempty, umlike

@cpptype(n_node=np.int32, n_octile=np.int32, p_degree=np.uintp,
         p_node=np.uintp, p_octile=np.uintp)
class OctileGraph:
    """ Python counterpart of C++ class graphdot::graph_t """

    @cpptype(elements=np.uintp, nzmask='<u8', nzmask_r='<u8',
             upper=np.int32, left=np.int32)
    class Octile:
        def __init__(self, upper, left, nzmask, nzmask_r, elements):
            self.upper = upper
            self.left = left
            self.nzmask = nzmask
            self.nzmask_r = nzmask_r
            self.elements = elements

    class CustomType:
        @cpptype(ptr=np.intp, size=np.int32)
        class FrozenArray(np.ndarray):
            @property
            def ptr(self):
                return ctypes.addressof(ctypes.c_char.from_buffer(self.base))

            @property
            def size(self):
                return len(self)

    def __init__(self, graph):

        self.nodes = nodes = graph.nodes.copy(deep=False)
        self.edges = edges = graph.edges.copy(deep=False)
        self.n_node = len(nodes)
        nnz = len(edges)

        ''' substitute columns corresponding to object-type node/edge 
        attributes to their GPU counterparts '''
        for df in [nodes, edges]:
            for key in list(df.columns):
                if not np.issctype(df[key].dtype):
                    attr_type = common_min_type(df[key], coerce=False)
                    if attr_type in [list, tuple]:
                        elem_type = common_min_type(
                            it.chain.from_iterable(df[key])
                        )
                        buffer = memoryview(umlike(np.fromiter(
                            it.chain.from_iterable(df[key]),
                            dtype=elem_type
                        )))
                        size = np.fromiter(map(len, df[key]), dtype=np.int)
                        head = np.cumsum(size) - size
                        # mangle key with type information
                        tag = '{key}::frozenarray::{dtype}'.format(
                            key=key,
                            dtype=json.dumps(elem_type.descr)
                        )
                        df[tag] = np.empty_like(df[key], dtype=np.object)
                        for i, (h, s) in enumerate(zip(head, size)):
                            df[tag][i] = np.frombuffer(
                                buffer[h:h + s], dtype=elem_type
                            ).view(self.CustomType.FrozenArray)
                        df.drop([key])
                        print(f'df[{tag}].dtype ', df[tag].dtype)
                    else:
                        raise TypeError(
                            f'Unsupported non-scalar attribute {key} '
                            f'of type {attr_type}'
                        )

        ''' add phantom label if none exists to facilitate C++ interop '''
        assert(len(nodes) >= 1)
        if len(nodes) == 1:
            nodes['labeled'] = np.zeros(len(nodes), np.bool_)

        assert(len(edges) >= 2)
        if len(edges) == 2:
            assert('!i' in edges and '!j' in edges)
            edges['labeled'] = np.zeros(len(edges), np.bool_)

        ''' determine node type '''
        self.node_type = node_type = nodes.rowtype(exclude=['!i'])
        print('node_type', node_type)
        self.node = umempty(len(nodes), dtype=node_type)

        def stateiter(df, types):
            for row in zip(*[df[key] for key in types.names]):
                yield tuple(i if np.isscalar(i) else i.state for i in row)
        print('list(stateiter(nodes, node_type))\n', list(stateiter(nodes, node_type)), sep='')
        print('self.node.dtype', self.node.dtype)
        self.node[:] = list(stateiter(nodes, node_type))
        # self.node[:] = list(zip(*[nodes[key] for key in node_type.names]))
        print('self.node\n', self.node, sep='')

        ''' determine whether graph is weighted, determine edge type,
            and compute node degrees '''
        self.degree = degree = umzeros(self.n_node, dtype=np.float32)
        edge_label_type = edges.rowtype(exclude=['!i', '!j', '!w'])
        if '!w' in edges:  # weighted graph
            self.weighted = True
            edge_type = np.dtype([('weight', np.float32),
                                  ('label', edge_label_type)], align=True)
            self.edge_type = edge_type
            np.add.at(degree, edges['!i'], edges['!w'])
            np.add.at(degree, edges['!j'], edges['!w'])

            if edge_label_type.itemsize != 0:
                labels = zip(*[edges[attr_type] for attr_type in edge_label_type.names])
            else:
                labels = [None] * len(edges)
            edge_aos = np.fromiter(zip(edges['!w'], labels), dtype=edge_type,
                                   count=nnz)
        else:
            self.weighted = False
            self.edge_type = edge_type = edge_label_type
            np.add.at(degree, edges['!i'], 1.0)
            np.add.at(degree, edges['!j'], 1.0)
            edge_aos = np.fromiter(zip(*[edges[attr_type] for attr_type in edge_type.names]),
                                   dtype=edge_type, count=nnz)
        degree[degree == 0] = 1.0

        ''' collect non-zero edge octiles '''
        indices = np.empty((4, nnz * 2), dtype=np.uint32, order='C')
        i, j, up, lf = indices
        i[:nnz] = edges['!i']
        j[:nnz] = edges['!j']
        # replicate & swap i and j for the lower triangular part
        indices[[0, 1], nnz:] = indices[[1, 0], :nnz]
        # get upper left corner of owner octiles
        up[:] = i - i % 8
        lf[:] = j - j % 8

        perm = np.lexsort(indices, axis=0)
        self.edge_aos = umempty(nnz * 2, edge_type)
        self.edge_aos[:] = edge_aos[perm % nnz]  # mod nnz due to symmetry

        diff = np.empty(nnz * 2)
        diff[1:] = (up[:-1] != up[1:]) | (lf[:-1] != lf[1:])
        diff[:1] = True
        oct_offset = np.flatnonzero(diff)
        self.n_octile = len(oct_offset)

        nzmasks = np.bitwise_or.reduceat(
            1 << (i - up + (j - lf) * 8).astype(np.uint64), oct_offset)
        nzmasks_r = np.bitwise_or.reduceat(
            1 << (j - lf + (i - up) * 8).astype(np.uint64), oct_offset)

        self.octiles = octiles = umempty(self.n_octile, self.Octile.dtype)
        octiles[:] = list(
            zip(int(self.edge_aos.base) + oct_offset * edge_type.itemsize,
                nzmasks,
                nzmasks_r,
                up[oct_offset],
                lf[oct_offset])
        )

    @property
    def p_octile(self):
        return int(self.octiles.base)

    @property
    def p_degree(self):
        return int(self.degree.base)

    @property
    def p_node(self):
        return int(self.node.base)
