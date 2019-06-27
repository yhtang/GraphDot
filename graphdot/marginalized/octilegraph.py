#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from pycuda.gpuarray import to_gpu
from graphdot.codegen.typetool import cpptype, rowtype


@cpptype([('upper', np.int32), ('left', np.int32),
          ('nzmask', np.uint64), ('elements', np.uintp)])
class Octile:
    """
    using octile_t = struct {
        int upper, left;
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


@cpptype([('n_node', np.int32), ('n_octile', np.int32),
          ('degree', np.uintp), ('node', np.uintp), ('octile', np.uintp)])
class OctileGraph:
    """
    struct graph_t {
        int n_node, n_octile;
        deg_t    * degree;
        node_t   * node;
        octile_t * octile;
    };
    """

    def __init__(self, graph, stopping_probability, wcol=None):
        ''' extract type information '''
        node_type = rowtype(graph.nodes)
        edge_type = rowtype(graph.edges.drop(['!ij'], axis=1))

        self.n_node = len(graph.nodes)

        ''' directly upload node labels to GPU '''
        node_d = to_gpu(graph.nodes[list(node_type.names)]
                        .to_records(index=False)
                        .astype(node_type))

        ''' node degree need to be computed from edges '''
        if wcol is None:
            # default label for edge weight lookup
            wcol = '!w'
        degree_h = np.zeros(self.padded_size, dtype=np.float32)
        if wcol in graph.edges.columns:
            for (i, j), w in zip(graph.edges['!ij'], graph.edges[wcol]):
                degree_h[i] += w
                degree_h[j] += w
        else:
            # treat as simple graph if no weights given
            degree_h[:self.n_node] = 1
        degree_h /= 1.0 - stopping_probability
        degree_d = to_gpu(degree_h)

        ''' collect non-zero octiles '''
        uniq_oct = np.unique([(i - i % 8, j - j % 8)
                              for i, j in graph.edges['!ij']], axis=0)
        uniq_oct = np.unique(uniq_oct + uniq_oct[:, -1::-1], axis=0)

        octile_table = {(upper, left): [0, np.zeros(64, dtype=edge_type)]
                        for upper, left in uniq_oct}

        for index, row in graph.edges.iterrows():
            i, j = row['!ij']
            edge = tuple(row[key] for key in edge_type.names)
            r = i % 8
            c = j % 8
            upper = i - r
            left = j - c
            octile_table[(upper, left)][0] |= 1 << (r + c * 8)
            octile_table[(upper, left)][1][r + c * 8] = edge
            octile_table[(left, upper)][0] |= 1 << (c + r * 8)
            octile_table[(left, upper)][1][c + r * 8] = edge

        ''' create octiles on GPU '''
        octile_list = [Octile(upper, left, nzmask, elements)
                       for (upper, left), (nzmask, elements) in octile_table.items()]

        ''' collect octile structures into continuous buffer '''
        octile_hdr = to_gpu(np.array([x.state for x in octile_list],
                                     Octile.dtype))

        self.node_type = node_type
        self.edge_type = edge_type

        self.n_octile = len(octile_list)
        self.__degree = degree_d
        self.__node = node_d
        self.__octile = octile_hdr
        self.octile_list = octile_list  # prevent automatic deconstruction

    @property
    def octile(self):
        return self.__octile.ptr

    @property
    def degree(self):
        return self.__degree.ptr

    @property
    def node(self):
        return self.__node.ptr

    @property
    def padded_size(self):
        return (self.n_node + 7) & ~7


if __name__ == '__main__':

    import pycuda.autoinit
    import networkx as nx
    import numpy as np
    from pycuda.compiler import SourceModule
    from graphdot.codegen import Template
    from graphdot.codegen.dtype import decltype

    from graphdot import Graph

    class Hybrid:
        NONE = 0
        SP = 1
        SP2 = 2
        SP3 = 3

    g = nx.Graph(title='H2O')
    g.add_node('O1', hybridization=Hybrid.SP2, charge=1, conjugate=False)
    g.add_node('H1', hybridization=Hybrid.SP3, charge=-1, conjugate=True)
    g.add_node('H2', hybridization=Hybrid.SP, charge=2, conjugate=True)
    # g.add_node('H2', hybridization=Hybrid.SP, charge=2, time=1)
    g.add_edge('O1', 'H1', order=1, length=0.5)
    g.add_edge('O1', 'H2', order=2, length=1.0)

    dfg = Graph.from_networkx(g)
    print(dfg.nodes)

    og = OctileGraph(dfg, 0.5)

    og_hdr = to_gpu(np.array([x.state for x in [og, og, og]], OctileGraph.dtype))

    mod = SourceModule(Template(r'''
    #include <cstdio>
    #include <cstdint>

    using bool_ = bool;
    using int_ = long;
    using intc = int;
    using intp = std::size_t;
    using uint8 = std::uint8_t;
    using uint16 = std::uint16_t;
    using uint32 = std::uint32_t;
    using uint64 = std::uint64_t;
    using int8 = std::int8_t;
    using int16 = std::int16_t;
    using int32 = std::int32_t;
    using int64 = std::int64_t;
    using float_ = double;
    using float32 = float;
    using float64 = double;

    struct graph_t {
        using node_t = ${node_t};
        using edge_t = ${edge_t};
        using octile_t = struct {
            int upper, left;
            edge_t * elements;
        };

        int n_node, n_octile;
        float    * degree;
        node_t   * node;
        octile_t * octile;
    };

    __global__ void fun(graph_t * graph_list, const int n_graph) {
        for(int I = 0; I < n_graph; ++I) {
            printf("Graph %d\n", I);
            auto & g = graph_list[I];
            printf("n_node %d\n", g.n_node);
            printf("n_octile %d\n", g.n_octile);
            for(int i = 0; i < g.n_node; ++i) {
                printf("node %d degree %f label (%ld, %ld, %d)\n", i, g.degree[i], g.node[i].hybridization, g.node[i].charge, g.node[i].conjugate);
            }
            for(int i = 0; i < g.n_octile; ++i) {
                printf("octile %d: (%d, %d)\n", i, g.octile[i].upper, g.octile[i].left);
                for(int r = 0; r < 8; ++r) {
                    for(int c = 0; c < 8; ++c) {
                        printf("(%ld,%.3lf) ", g.octile[i].elements[r + c * 8].order, g.octile[i].elements[r + c * 8].length);
                    }
                    printf("\n");
                }
            }
        }
    }
    ''').render(node_t=decltype(rowtype(dfg.nodes)),
                edge_t=decltype(rowtype(dfg.edges.drop(['!ij'], axis=1)))))

    fun = mod.get_function('fun')

    fun(og_hdr, np.int32(3), grid=(1,1,1), block=(1,1,1))
