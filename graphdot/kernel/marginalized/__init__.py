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
from ._scratch import BlockScratch
from ._octilegraph import OctileGraph
from .basekernel import TensorProduct, _Multiply

__all__ = ['MarginalizedGraphKernel']

try:
    pycuda.driver.init()
except Exception as e:
    raise RuntimeWarning('PyCUDA initialization failed, message: ' + str(e))


# only works with python >= 3.6
# @cpptype(i=np.int32, j=np.int32)
@cpptype([('i', np.int32), ('j', np.int32)])
class JobIn:
    def __init__(self, i, j):
        self.i = i
        self.j = j


# only works with python >= 3.6
# @cpptype(similarity=np.float32, iterations=np.int32)
@cpptype([('similarity', np.float32), ('iterations', np.int32)])
class JobOut:
    pass


class MarginalizedGraphKernel:
    """Implements the random walk-based graph similarity kernel as proposed in:
    Kashima, H., Tsuda, K., & Inokuchi, A. (2003).
    Marginalized kernels between labeled graphs. *In Proceedings of the 20th
    international conference on machine learning (ICML-03)* (pp. 321-328).

    Parameters
    ----------
    node_kernel: base kernel or composition of base kernels
        A kernelet that computes the similarity between individual nodes
    edge_kernel: base kernel or composition of base kernels
        A kernelet that computes the similarity between individual edge
    kwargs: optional arguments
        q: float in (0, 1)
            The probability for the random walk to stop during each step
        block_per_sm: int
            Tunes the GPU kernel
        block_size: int
            Tunes the GPU kernel
    """

    _template = os.path.join(os.path.dirname(__file__), 'template.cu')

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

    def clone_with_theta():
        """scikit-learn compatibility method"""
        pass

    def _assert_homegeneous(self, X):
        for x1, x2 in zip(X[:-1], X[1:]):
            try:
                assert(x1.weighted == x2.weighted)
                assert(x1.node_type == x2.node_type)
                assert(x1.edge_type == x2.edge_type)
            except AssertionError as e:
                raise TypeError('All graphs must be of the same type: %s' %
                                str(e))

    def __call__(self, X, Y=None):
        """Compute pairwise similarity matrix between graphs

        Parameters
        ----------
        X: list of N graphs
            The graphs must all have same node and edge attributes.
        Y: None or list of M graphs
            The graphs must all have same node and edge attributes.

        Returns
        -------
        numpy.array
            if Y is None, return a N-by-N matrix containing pairwise
            similarities between the graphs in X; otherwise, returns a N-by-M
            matrix containing similarities across graphs in X and Y.
        """

        ''' transfer grahs to GPU '''
        X = [OctileGraph(x) for x in X]
        Y = [OctileGraph(y) for y in Y] if Y is not None else []
        self._assert_homegeneous(X + Y)
        graph_list_d = to_gpu(np.array([g.state for g in X + Y],
                                       OctileGraph.dtype))

        ''' prepare pairwise work item list '''
        N = len(X)
        M = len(Y)
        if len(Y):
            jobs = np.array([JobIn(i, N + j).state
                             for i in range(N)
                             for j in range(M)], JobIn.dtype)
        else:
            jobs = np.array([JobIn(i, j).state
                             for i in range(N)
                             for j in range(i, N)], JobIn.dtype)
        jobs_d = to_gpu(jobs)
        i_job_global = pycuda.gpuarray.zeros(1, np.uint32)

        ''' prepare GPU kernel launch '''
        x = next(iter(X))
        weighted = x.weighted
        node_type = x.node_type
        edge_type = x.edge_type

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

        launch_block_count = (self.device.MULTIPROCESSOR_COUNT
                              * self.block_per_sm)
        shmem_bytes_per_warp = mod.get_global('shmem_bytes_per_warp')[1]
        shmem_bytes_per_block = (shmem_bytes_per_warp * self.block_size
                                 // self.device.WARP_SIZE)

        max_graph_size = np.max([g.padded_size for g in X + Y])
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

        ''' collect result '''
        result = jobs_d.get().view(JobOut.dtype)

        if len(Y):
            R = np.zeros((N, M))
            for (i, j), (r, iteration) in zip(jobs, result):
                R[i, j - N] = r
        else:
            R = np.zeros((N, N))
            for (i, j), (r, iteration) in zip(jobs, result):
                R[i, j] = R[j, i] = r

        return R
