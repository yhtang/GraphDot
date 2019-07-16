#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import numpy as np
import pycuda.driver
import pycuda.gpuarray
from pycuda.compiler import SourceModule
from pycuda.gpuarray import to_gpu
from graphdot import cpp
from graphdot.codegen import Template
from graphdot.codegen.typetool import cpptype, decltype
from .scratch import BlockScratch
from .octilegraph import OctileGraph
from .basekernel import TensorProduct, _Multiply

__all__ = ['MarginalizedGraphKernel']

try:
    pycuda.driver.init()
except Exception as e:
    raise RuntimeWarning('PyCUDA initialization failed, message: ' + str(e))


# only works with python >= 3.6
# @cpptype(i=np.int32, j=np.int32)
@cpptype([('i', np.int32), ('j', np.int32)])
class JobIn(object):
    def __init__(self, i, j):
        self.i = i
        self.j = j


# only works with python >= 3.6
# @cpptype(similarity=np.float32, iterations=np.int32)
@cpptype([('similarity', np.float32), ('iterations', np.int32)])
class JobOut(object):
    pass


class MarginalizedGraphKernel(object):

    _template = os.path.join(os.path.dirname(__file__), 'kernel.cu')

    def __init__(self, node_kernel, edge_kernel, **kwargs):
        self.node_kernel = node_kernel
        self.edge_kernel = edge_kernel
        self.template = Template(self._template)
        self.scratch = None
        self.scratch_capacity = 0

        self.q = kwargs.pop('q', 0.01)

        self.block_per_sm = kwargs.pop('block_per_sm', 8)
        self.block_size = kwargs.pop('block_size', 128)

        self.device = pycuda.driver.Device(kwargs.pop('block_per_sm', 0))
        self.nvcc_extra = kwargs.pop('nvcc_extra', [])
        self.ctx = self.device.make_context()

    def __del__(self):
        self.ctx.synchronize()
        self.ctx.pop()

    def _allocate_scratch(self, count, capacity):
        if (self.scratch is None or len(self.scratch) < count or
                self.scratch[0].capacity < capacity):
            self.ctx.synchronize()
            self.scratch = [BlockScratch(capacity) for _ in range(count)]
            self.scratch_d = to_gpu(np.array([s.state for s in self.scratch],
                                             BlockScratch.dtype))
            self.scratch_capacity = self.scratch[0].capacity
            self.ctx.synchronize()

    def compute(self, graph_list):
        # TODO: graph registry
        graph_list = [OctileGraph(g) for g in graph_list]

        weighted = any([g.weighted for g in graph_list])

        if weighted:
            edge_kernel = TensorProduct(weight=_Multiply(),
                                        label=self.edge_kernel)
        else:
            edge_kernel = self.edge_kernel

        node_kernel_src = Template(r'''
        struct node_kernel {
            template<class V> __device__
            static auto compute(V const &v1, V const &v2) {
                return ${node_expr};
            }
        };
        ''').render(node_expr=self.node_kernel.gencode('v1', 'v2'))

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

        mod = SourceModule(source,
                           options=['-std=c++14',
                                    '-O4',
                                    '--use_fast_math',
                                    '--expt-relaxed-constexpr',
                                    '--maxrregcount=64',
                                    ] + self.nvcc_extra,
                           no_extern_c=True,
                           include_dirs=cpp.__path__)
        kernel = mod.get_function('graph_kernel_solver')

        N = len(graph_list)
        jobs = np.array([JobIn(i, j).state
                         for i in range(N)
                         for j in range(i, N)], JobIn.dtype)
        jobs_d = to_gpu(jobs)

        i_job_global = pycuda.gpuarray.zeros(1, np.uint32)

        graph_list_d = to_gpu(np.array([g.state for g in graph_list],
                                       OctileGraph.dtype))

        launch_block_count = (self.device.MULTIPROCESSOR_COUNT
                              * self.block_per_sm)
        shmem_bytes_per_warp = mod.get_global('shmem_bytes_per_warp')[1]
        shmem_bytes_per_block = (shmem_bytes_per_warp * self.block_size
                                 // self.device.WARP_SIZE)

        max_graph_size = np.max([g.padded_size for g in graph_list])
        self._allocate_scratch(launch_block_count, max_graph_size**2)

        # print("%-32s : %ld" % ("Blocks launched", launch_block_count))
        # print("%-32s : %ld" % ("Shared memory per block",
        #                        shmem_bytes_per_block))

        kernel(graph_list_d,
               self.scratch_d,
               jobs_d,
               i_job_global,
               np.uint32(jobs.size),
               np.float32(1.0),
               np.float32(self.q),
               grid=(launch_block_count, 1, 1),
               block=(self.block_size, 1, 1),
               shared=shmem_bytes_per_block)

        result = jobs_d.get().view(JobOut.dtype)

        R = np.zeros((N, N))
        for (i, j), (r, iter) in zip(jobs, result):
            R[i, j] = R[j, i] = r

        return R
