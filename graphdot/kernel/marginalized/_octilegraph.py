#!/usr/bin/env python
# -*- coding: utf-8 -*-
import ctypes
import itertools as it
import numpy as np
from graphdot.codegen.cpptool import cpptype
from graphdot.codegen.typetool import common_min_type
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

        ''' substitute columns corresponding to object-type node/edge
        attributes to their GPU counterparts '''
        for df in [nodes, edges]:
            for key in list(df.columns):
                if not np.issctype(df[key].dtype):
                    if issubclass(df[key].concrete_type, (list,
                                                          tuple,
                                                          np.ndarray)):
                        inner_type = common_min_type.of_types(
                            [x.dtype if isinstance(x, np.ndarray)
                             else common_min_type.of_values(x)
                             for x in df[key]]
                        )
                        if not np.issctype(inner_type):
                            raise(TypeError(
                                f'Expect scalar elements in tuple or list'
                                f'atttributes, got {inner_type}.'
                            ))
                        if not np.issctype(inner_type):
                            raise TypeError(
                                f'List-like graphs attribute must have scalar'
                                f'elements. Attribute {key} is {inner_type}.'
                            )
                        buffer = memoryview(umlike(np.fromiter(
                            it.chain.from_iterable(df[key]),
                            dtype=inner_type
                        )))
                        size = np.fromiter(map(len, df[key]), dtype=np.int)
                        head = np.cumsum(size) - size
                        # mangle key with type information
                        tag = '${key}::frozen_array::{dtype}'.format(
                            key=key,
                            dtype=inner_type.str
                        )
                        data = np.empty_like(df[key], dtype=np.object)
                        for i, (h, s) in enumerate(zip(head, size)):
                            data[i] = np.frombuffer(
                                buffer[h:h + s], dtype=inner_type
                            ).view(self.CustomType.FrozenArray)
                        df[tag] = data
                        df.drop([key], inplace=True)
                    else:
                        raise TypeError(
                            f'Unsupported non-scalar attribute {key} '
                            f'of type {df[key].concrete_type}'
                        )

        ''' add phantom label if none exists to facilitate C++ interop '''
        assert(len(nodes.columns) >= 1)
        if len(nodes.columns) == 1:
            nodes['labeled'] = np.zeros(len(nodes), np.bool_)

        assert(len(edges.columns) >= 2)
        if len(edges.columns) == 2:
            assert('!i' in edges and '!j' in edges)
            edges['labeled'] = np.zeros(len(edges), np.bool_)

        ''' determine node type '''
        i = nodes['!i']
        nodes.drop(['!i'], inplace=True)
        self.node_t = node_t = nodes.rowtype()
        self.nodes_aos = umempty(len(nodes), dtype=node_t)
        self.nodes_aos[i] = list(nodes.iterstates())

        ''' determine whether graph is weighted, determine edge type,
            and compute node degrees '''
        self.degree = degree = umzeros(self.n_node, dtype=np.float32)
        edge_t = edges.drop(['!i', '!j', '!w']).rowtype()
        self_loops = edges[edges['!i'] == edges['!j']]
        nnz = len(edges)
        if '!w' in edges:  # weighted graph
            self.weighted = True
            np.add.at(degree, edges['!i'], edges['!w'])
            np.add.at(degree, edges['!j'], edges['!w'])
            np.subtract.at(degree, self_loops['!i'], self_loops['!w'])

            if edge_t.itemsize != 0:
                labels = list(edges[edge_t.names].iterstates())
            else:
                labels = [None] * len(edges)

            edge_t = np.dtype(
                [('weight', np.float32), ('label', edge_t)],
                align=True
            )

            edges_aos = np.fromiter(zip(edges['!w'], labels), dtype=edge_t,
                                    count=nnz)
        else:
            self.weighted = False
            np.add.at(degree, edges['!i'], 1.0)
            np.add.at(degree, edges['!j'], 1.0)
            np.subtract.at(degree, self_loops['!i'], 1.0)
            edges_aos = np.fromiter(edges[edge_t.names].iterstates(),
                                    dtype=edge_t, count=nnz)
        self.edge_t = edge_t
        degree[degree == 0] = 1.0

        ''' collect non-zero edge octiles '''
        indices = np.empty((4, nnz * 2), dtype=np.uint32, order='C')
        i, j, up, lf = indices
        i[:nnz] = edges['!i']
        j[:nnz] = edges['!j']
        # replicate & swap i and j for the lower triangular part
        i[nnz:], j[nnz:] = j[:nnz], i[:nnz]
        # get upper left corner of owner octiles
        up[:] = i - i % 8
        lf[:] = j - j % 8

        # np.unique implies lexical sort
        (lf, up, j, i), perm = np.unique(
            indices[-1::-1, :], axis=1, return_index=True
        )
        self.edges_aos = umempty(len(i), edge_t)
        self.edges_aos[:] = edges_aos[perm % nnz]  # mod nnz due to symmetry

        diff = np.empty_like(up)
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
            zip(int(self.edges_aos.base) + oct_offset * edge_t.itemsize,
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
        return int(self.nodes_aos.base)
