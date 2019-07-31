#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import uuid
import numpy as np
import pycuda.driver
import pycuda.gpuarray
from pycuda.compiler import SourceModule
from pycuda.gpuarray import to_gpu, GPUArray
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
@cpptype([('i', np.int32), ('j', np.int32), ('vr', np.uintp)])
class Job:
    def __init__(self, i, j, vr_gpu):
        self.i = i
        self.j = j
        self.vr_gpu = vr_gpu

    @property
    def vr(self):
        return self.vr_gpu.ptr


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
        p: functor or 'uniform' or 'default'
            The starting probability of the random walk on each node. Must be
            either a functor that takes in a node (a dataframe row) and returns
            a number, or the name of a built-in distribution. Currently, only
            'uniform' and 'default' are implemented.
            Note that a custom probability does not have to be normalized.
        q: float in (0, 1)
            The probability for the random walk to stop during each step.
        block_per_sm: int
            Tunes the GPU kernel.
        block_size: int
            Tunes the GPU kernel.
    """

    _template = os.path.join(os.path.dirname(__file__), 'template.cu')

    def __init__(self, node_kernel, edge_kernel, **kwargs):
        self.node_kernel = node_kernel
        self.edge_kernel = edge_kernel
        self.template = Template(self._template)
        self.scratch = None
        self.scratch_capacity = 0
        self.graph_cache = {}

        self.p = kwargs.pop('p', 'default')
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

    def _convert_to_octilegraph(self, graph):
        if hasattr(graph, 'uuid') and graph.uuid in self.graph_cache:
            return self.graph_cache[graph.uuid]
        else:
            if not hasattr(graph, 'uuid'):
                graph.uuid = uuid.uuid4()
            og = OctileGraph(graph)
            self.graph_cache[graph.uuid] = og
            return og

    def __call__(self, X, Y=None, nodal=False):
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

        ''' assign starting probabilities '''
        if isinstance(self.p, str):
            if self.p == 'uniform' or self.p == 'default':
                pX = [np.ones(len(g.nodes)) for g in X]
                pY = [np.ones(len(g.nodes)) for g in Y] if Y else []
            else:
                raise ValueError('Unknown starting probability distribution %s'
                                 % self.p)
        else:
            pX = [np.array([self.p(node) for node in g.nodes.iterrows()])
                  for g in X]
            pY = [np.array([self.p(node) for node in g.nodes.iterrows()])
                  for g in Y] if Y else []

        ''' transfer grahs to GPU '''
        X = [self._convert_to_octilegraph(x) for x in X]
        Y = [self._convert_to_octilegraph(y) for y in Y] if Y else []
        self._assert_homegeneous(X + Y)
        graph_list_d = to_gpu(np.array([g.state for g in X + Y],
                                       OctileGraph.dtype))

        ''' prepare pairwise work item list '''
        N = len(X)
        M = len(Y)
        if len(Y):
            jobs = [Job(i, N + j, GPUArray(g1.n_node * g2.n_node, np.float32))
                    for i, g1 in enumerate(X)
                    for j, g2 in enumerate(Y)]
        else:
            jobs = [Job(i, i + j, GPUArray(g1.n_node * g2.n_node, np.float32))
                    for i, g1 in enumerate(X)
                    for j, g2 in enumerate(X[i:])]
        jobs_d = to_gpu(np.array([j.state for j in jobs], Job.dtype))
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
               np.uint32(len(jobs)),
               np.float32(self.q),
               np.float32(self.q),  # placeholder for q0
               grid=(launch_block_count, 1, 1),
               block=(self.block_size, 1, 1),
               shared=shmem_bytes_per_block)

        ''' collect result '''
        if len(Y):
            R = np.empty((N, M), np.object)
            for job in jobs:
                r = job.vr_gpu.get().reshape(X[job.i].n_node, -1)
                pi = pX[job.i]
                pj = pY[job.j - N]
                if nodal is True:
                    R[job.i, job.j - N] = pi[:, None] * r * pj[None, :]
                else:
                    R[job.i, job.j - N] = pi.dot(r).dot(pj)
        else:
            R = np.empty((N, N), np.object)
            for job in jobs:
                r = job.vr_gpu.get().reshape(X[job.i].n_node, -1)
                pi = pX[job.i]
                pj = pX[job.j]
                if nodal is True:
                    R[job.i, job.j] = pi[:, None] * r * pj[None, :]
                    R[job.j, job.i] = R[job.i, job.j].T
                else:
                    R[job.i, job.j] = R[job.j, job.i] = pi.dot(r).dot(pj)

        return np.block(R.tolist())
