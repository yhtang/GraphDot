#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest
import numpy as np
import pycuda.autoinit
from graphdot.kernel.marginalized._octilegraph import OctileGraph
from graphdot import Graph


def test_octile_graph_unweighted():

    assert(OctileGraph.dtype.isalignedstruct)

    dfg = Graph(
        nodes={
            '!i': [0, 1, 2],
            'charge': [1, -1, 2],
            'conjugate': [False, True, True],
            'hybridization': [2, 3, 1]
        },
        edges={
            '!i': [0, 0],
            '!j': [1, 2],
            'length': [0.5, 1.0],
            'order': [1, 2]
        },
        title='H2O')

    og = OctileGraph(dfg)
    assert(og.n_node == len(dfg.nodes))
    assert(og.p_octile != 0)
    assert(og.p_degree != 0)
    assert(og.p_node != 0)
    with pytest.raises(AttributeError):
        og.p_octile = np.uintp(0)
    with pytest.raises(AttributeError):
        og.p_degree = np.uintp(0)
    with pytest.raises(AttributeError):
        og.p_node = np.uintp(0)

    assert(og.node_type.isalignedstruct)
    for name in og.node_type.names:
        assert(name in dfg.nodes.columns)
    assert('charge' in og.node_type.names)
    assert('conjugate' in og.node_type.names)
    assert('hybridization' in og.node_type.names)

    assert(og.edge_type.isalignedstruct)
    for name in og.edge_type.names:
        assert(name in dfg.edges.columns)
    for name in dfg.edges.columns:
        if name in ['!i', '!j']:
            continue
        assert(name in og.edge_type.names)


def test_octile_graph_weighted():

    assert(OctileGraph.dtype.isalignedstruct)

    dfg = Graph(
        nodes={
            '!i': [0, 1, 2],
            'charge': [1, -1, 2],
            'conjugate': [False, True, True],
            'hybridization': [2, 3, 1]
        },
        edges={
            '!i': [0, 0],
            '!j': [1, 2],
            'length': [0.5, 1.0],
            '!w': [1.0, 2.0]
        },
        title='H2O')

    og = OctileGraph(dfg)
    assert(og.n_node == len(dfg.nodes))
    assert(og.p_octile != 0)
    assert(og.p_degree != 0)
    assert(og.p_node != 0)
    with pytest.raises(AttributeError):
        og.p_octile = np.uintp(0)
    with pytest.raises(AttributeError):
        og.p_degree = np.uintp(0)
    with pytest.raises(AttributeError):
        og.p_node = np.uintp(0)

    assert(og.node_type.isalignedstruct)
    for name in og.node_type.names:
        assert(name in dfg.nodes.columns)
    assert('charge' in og.node_type.names)
    assert('conjugate' in og.node_type.names)
    assert('hybridization' in og.node_type.names)

    assert(og.edge_type.isalignedstruct)
    assert(len(og.edge_type.names) == 2)
    assert('weight' in og.edge_type.names)
    assert('label' in og.edge_type.names)

    for name in og.edge_type['label'].names:
        assert(name in dfg.edges.columns)
    for name in dfg.edges.columns:
        if name in ['!i', '!j', '!w']:
            continue
        assert(name in og.edge_type['label'].names)

    # from pycuda.compiler import SourceModule
    # from graphdot.codegen import Template
    # from graphdot.codegen.typetool import decltype

    # og_hdr = to_gpu(np.array([x.state for x in [og, og, og]],
    #                          OctileGraph.dtype))
    #
    # mod = SourceModule(Template(r'''
    # #include <cstdio>
    # #include <cstdint>
    #
    # using bool_ = bool;
    # using int_ = long;
    # using intc = int;
    # using intp = std::size_t;
    # using uint8 = std::uint8_t;
    # using uint16 = std::uint16_t;
    # using uint32 = std::uint32_t;
    # using uint64 = std::uint64_t;
    # using int8 = std::int8_t;
    # using int16 = std::int16_t;
    # using int32 = std::int32_t;
    # using int64 = std::int64_t;
    # using float_ = double;
    # using float32 = float;
    # using float64 = double;
    #
    # struct graph_t {
    #     using node_t = ${node_t};
    #     using edge_t = ${edge_t};
    #     using octile_t = struct {
    #         int upper, left;
    #         edge_t * elements;
    #     };
    #
    #     int n_node, n_octile;
    #     float    * degree;
    #     node_t   * node;
    #     octile_t * octile;
    # };
    #
    # __global__ void fun(graph_t * graph_list, const int n_graph) {
    #     for(int I = 0; I < n_graph; ++I) {
    #         printf("Graph %d\n", I);
    #         auto & g = graph_list[I];
    #         printf("n_node %d\n", g.n_node);
    #         printf("n_octile %d\n", g.n_octile);
    #         for(int i = 0; i < g.n_node; ++i) {
    #             printf("node %d degree %f label (%ld, %ld, %d)\n",
    #             i, g.degree[i], g.node[i].hybridization,
    #             g.node[i].charge, g.node[i].conjugate);
    #         }
    #         for(int i = 0; i < g.n_octile; ++i) {
    #             printf("octile %d: (%d, %d)\n",
    #                    i, g.octile[i].upper, g.octile[i].left);
    #             for(int r = 0; r < 8; ++r) {
    #                 for(int c = 0; c < 8; ++c) {
    #                     printf("(%ld,%.3lf) ",
    #                            g.octile[i].elements[r + c * 8].order,
    #                            g.octile[i].elements[r + c * 8].length);
    #                 }
    #                 printf("\n");
    #             }
    #         }
    #     }
    # }
    # ''').render(node_t=decltype(rowtype(dfg.nodes)),
    #             edge_t=decltype(rowtype(dfg.edges.drop(['!ij'], axis=1)))))
    #
    # fun = mod.get_function('fun')
    #
    # fun(og_hdr, np.int32(3), grid=(1, 1, 1), block=(1, 1, 1))
