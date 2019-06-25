#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
sys.path.append('/home/ytang/Seafile/research/source/graphdot')


import os
import numpy
import pycuda
import pycuda.gpuarray
from pycuda.compiler import SourceModule
from graphdot.codegen import Template
from graphdot.codegen.dtype import decltype
import graphdot.cpp


class MarginalizedGraphKernel:

    _template = os.path.join(os.path.dirname(__file__), 'kernel.cu')

    def __init__(self, node_kernel, edge_kernel):
        self.node_kernel = node_kernel
        self.edge_kernel = edge_kernel
        self.template = Template(self._template)

    @classmethod
    def _pack_type(cls, df):
        order = numpy.argsort([df.dtypes[key].itemsize for key in df.columns])
        packed_attributes = [df.columns[i] for i in order[-1::-1]]
        packed_dtype = numpy.dtype([(key, df.dtypes[key].newbyteorder('='))
                                    for key in packed_attributes], align=True)
        return packed_dtype

    def compute(self, graph):
        node_type = self._pack_type(graph.nodes)
        edge_type = self._pack_type(graph.edges.drop(['_ij'], axis=1))

        node_kernel_src = Template(r'''
        struct node_kernel {
            template<class V> __device__
            static auto compute(V const &v1, V const &v2) {
                return ${node_expr};
            }
        };
        ''').render(node_expr=node_kernel.gencode('v1', 'v2'))

        edge_kernel_src = Template(r'''
        struct edge_kernel {
            template<class T> __device__
            static auto compute(T const &e1, T const &e2) {
                return ${edge_expr};
            }
        };
        ''').render(edge_expr=edge_kernel.gencode('e1', 'e2'))

        source = self.template.render(node_kernel=node_kernel_src,
                                      edge_kernel=edge_kernel_src,
                                      node_t=decltype(node_type),
                                      edge_t=decltype(edge_type))

        print('SOURCE\n', source, sep='')

        mod = SourceModule(source,
                           options=['-std=c++14',
                                    '--expt-relaxed-constexpr'],
                           no_extern_c=True,
                           include_dirs=graphdot.cpp.__path__)
        print(mod)
        print(mod.get_function('graph_kernel_solver'))
        # node_gpu = pycuda.gpuarray.GPUArray(df.shape[0], packed_dtype)
        # print(repr(node_gpu))


if __name__ == '__main__':

    import pycuda.autoinit

    if True:
        import networkx as nx

        # from graphdot.marginalized.basekernel import Constant
        from graphdot.marginalized.basekernel import KroneckerDelta
        from graphdot.marginalized.basekernel import SquareExponential
        from graphdot.marginalized.basekernel import KeywordTensorProduct
        # from graphdot.marginalized.basekernel import Convolution

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

        gg = Graph.from_networkx(g)

        print('GRAPH:\n', gg)

        node_kernel = KeywordTensorProduct(
                          hybridization=KroneckerDelta(0.3, 1.0),
                          charge=SquareExponential(1.0),
                          conjugate=KroneckerDelta(0.5))

        edge_kernel = KeywordTensorProduct(
                          order=KroneckerDelta(0.3, 1.0),
                          length=SquareExponential(0.05))

        mlgk = MarginalizedGraphKernel(node_kernel, edge_kernel)

        mlgk.compute(gg)
