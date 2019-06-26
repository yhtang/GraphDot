#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import numpy as np
import pycuda.driver
import pycuda.gpuarray
from pycuda.compiler import SourceModule
from pycuda.gpuarray import to_gpu
from graphdot.codegen import Template
from graphdot.codegen.dtype import decltype
from graphdot.codegen.interop import cpptype
from graphdot.marginalized.scratch import BlockScratch
from graphdot.marginalized.octilegraph import OctileGraph
import graphdot.cpp


pycuda.driver.init()


@cpptype([('i', np.int32), ('j', np.int32)])
class JobIn:
    def __init__(self, i, j):
        self.i = i
        self.j = j


# @cpptype([('i', np.int32), ('j', np.int32)])
# class JobOut:
#     def __init__(self, i, j):
#         self.i = i
#         self.j = j


class MarginalizedGraphKernel:

    _template = os.path.join(os.path.dirname(__file__), 'kernel.cu')

    def __init__(self, node_kernel, edge_kernel, **kwargs):
        self.node_kernel = node_kernel
        self.edge_kernel = edge_kernel
        self.template = Template(self._template)
        self.scratch = None
        self.scratch_capacity = 0

        self.q = kwargs.pop('q', 0.05)

        self.block_per_sm = kwargs.pop('block_per_sm', 8)
        self.block_size = kwargs.pop('block_size', 128)

        self.device = pycuda.driver.Device(kwargs.pop('block_per_sm', 0))
        self.ctx = self.device.make_context()

    def __del__(self):
        self.ctx.synchronize()
        self.ctx.pop()

    def _allocate_scratch(self, count, capacity):
        if (self.scratch is None or
                len(self.scratch) < count or
                self.scratch[0].capacity < capacity):
            self.ctx.synchronize()
            self.scratch = [BlockScratch(capacity) for _ in range(count)]
            self.scratch_d = to_gpu(np.array([s.state for s in self.scratch],
                                             BlockScratch.dtype))
            self.scratch_capacity = self.scratch[0].capacity
            self.ctx.synchronize()

    def compute(self, graph_list):
        # TODO: graph registry
        graph_list = [OctileGraph(g, 0.5) for g in graph_list]

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

        node_type = graph_list[0].node_type
        edge_type = graph_list[0].edge_type

        source = self.template.render(node_kernel=node_kernel_src,
                                      edge_kernel=edge_kernel_src,
                                      node_t=decltype(node_type),
                                      edge_t=decltype(edge_type))

        print('SOURCE\n', source, sep='')

        mod = SourceModule(source,
                           options=['-std=c++14',
                                    '-O4',
                                    '--use_fast_math',
                                    '--maxrregcount=64',
                                    '-Xptxas', '-v',
                                    '--expt-relaxed-constexpr'],
                           no_extern_c=True,
                           include_dirs=graphdot.cpp.__path__)
        kernel = mod.get_function('graph_kernel_solver')
        print('KERNEL\n', kernel)
        print(kernel.num_regs)

        N = len(graph_list)
        jobs = to_gpu(np.array([JobIn(i, j).state
                                for i in range(N)
                                for j in range(i, N)],
                               JobIn.dtype))

        i_job_global = pycuda.gpuarray.zeros(1, np.uint32)

        graph_list_d = to_gpu(np.array([g.state for g in graph_list],
                                       OctileGraph.dtype))

        launch_block_count = self.device.MULTIPROCESSOR_COUNT * self.block_per_sm
        shmem_bytes_per_warp = mod.get_global('shmem_bytes_per_warp')[1]
        shmem_bytes_per_block = shmem_bytes_per_warp * self.block_size // self.device.WARP_SIZE

        max_graph_size = np.max([g.padded_size for g in graph_list])
        self._allocate_scratch(launch_block_count, max_graph_size * max_graph_size)

        print("%-32s : %ld" % ("Blocks launched", launch_block_count))
        print("%-32s : %ld" % ("Shared memory per block", shmem_bytes_per_block))

        kernel(graph_list_d,
               self.scratch_d,
               jobs,
               i_job_global,
               np.uint32(jobs.size),
               np.float32(1.0),
               np.float32(self.q),
               grid=(launch_block_count, 1, 1),
               block=(self.block_size, 1, 1),
               shared=shmem_bytes_per_block)

    #     cuda::sync_and_peek( __FILE__, __LINE__ );
    #
    #     cudaMemcpyAsync( job_list_cpu.data(), dev_jobs, dev_jobs.size * dev_jobs.element_size, cudaMemcpyDefault );
    #
    #     cuda::verify( ( cudaDeviceSynchronize() ) );
    #
    #     std::vector<float> result;
    #     for(auto const &j: job_list_cpu) result.push_back( j.out.r );
    #
    #     return result;
    # }

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

        g1 = nx.Graph(title='H2O')
        g1.add_node('O1', hybridization=Hybrid.SP2, charge=1, conjugate=False)
        g1.add_node('H1', hybridization=Hybrid.SP3, charge=-1, conjugate=True)
        g1.add_node('H2', hybridization=Hybrid.SP, charge=2, conjugate=True)
        # g.add_node('H2', hybridization=Hybrid.SP, charge=2, time=1)
        g1.add_edge('O1', 'H1', order=1, length=0.5)
        g1.add_edge('O1', 'H2', order=2, length=1.0)

        g2 = nx.Graph(title='H2')
        g2.add_node('H1', hybridization=Hybrid.SP, charge=1, conjugate=True)
        g2.add_node('H2', hybridization=Hybrid.SP, charge=1, conjugate=True)
        g2.add_edge('H1', 'H2', order=2, length=1.0)

        node_kernel = KeywordTensorProduct(
                          hybridization=KroneckerDelta(0.3, 1.0),
                          charge=SquareExponential(1.0),
                          conjugate=KroneckerDelta(0.5))

        edge_kernel = KeywordTensorProduct(
                          order=KroneckerDelta(0.3, 1.0),
                          length=SquareExponential(0.05))

        mlgk = MarginalizedGraphKernel(node_kernel, edge_kernel)

        mlgk._allocate_scratch(10, 117)
        print(mlgk.scratch_capacity)
        print(mlgk.scratch_d)
        print(BlockScratch.dtype)

        mlgk.compute([Graph.from_networkx(g1), Graph.from_networkx(g2)])
