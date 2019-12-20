#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import copy
import uuid
import warnings
import functools
import numpy as np
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import graphdot.cuda
from graphdot import cpp
from graphdot.codegen import Template
from graphdot.codegen.typetool import decltype
from graphdot.cuda.array import umempty, umzeros, umarray, umlike
from graphdot.cuda.resizable_array import ResizableArray
from graphdot.kernel.basekernel import TensorProduct, _Multiply
from ._backend import Backend
from ._scratch import BlockScratch
from ._octilegraph import OctileGraph


class CUDABackend(Backend):

    @staticmethod
    def array(ndarray):
        return umarray(ndarray)

    @staticmethod
    def zeros(size, dtype=np.float32):
        return umzeros(size, dtype)

    @staticmethod
    def empty(size, dtype=np.float32):
        return umempty(size, dtype)

    def __init__(self, **kwargs):
        self.ctx = kwargs.pop('cuda_context', graphdot.cuda.defctx)
        self.device = self.ctx.get_device()
        self.scratch = None
        self.scratch_capacity = 0
        self.graph_cache = {}
        self.graph_cpp = ResizableArray(OctileGraph.dtype, allocator='numpy')

        self.block_per_sm = kwargs.pop('block_per_sm', 8)
        self.block_size = kwargs.pop('block_size', 128)

        self.nvcc_extra = kwargs.pop('nvcc_extra', [])
        self._source = ''
        self._module = None

    def __deepcopy__(self, memo):
        return copy.copy(self)

    def _assert_homogeneous(self, x, y):
        try:
            assert(x.weighted == y.weighted)
            assert(x.node_type == y.node_type)
            assert(x.edge_type == y.edge_type)
        except AssertionError as e:
            raise TypeError('All graphs must be of the same type: %s' %
                            str(e))

    def _allocate_scratch(self, count, capacity):
        if (self.scratch is None or len(self.scratch) < count or
                self.scratch[0].capacity < capacity):
            self.ctx.synchronize()
            self.scratch = [BlockScratch(capacity) for _ in range(count)]
            self.scratch_d = umarray(np.array([s.state for s in self.scratch],
                                              BlockScratch.dtype))
            self.scratch_capacity = self.scratch[0].capacity
            self.ctx.synchronize()

    def _register_graph(self, graph, p):
        if not hasattr(graph, 'uuid'):
            graph.uuid = uuid.uuid4()
        if graph.uuid not in self.graph_cache:
            # convert to GPU format
            og = OctileGraph(graph)
            # assign starting probabilities
            ps = umarray(np.array([p(*r) for r in graph.nodes.iterrows()],
                                  dtype=np.float32))
            i = len(self.graph_cpp)
            self.graph_cache[graph.uuid] = (i, ps, og)
            self.graph_cpp.append(og.state)
        return self.graph_cache[graph.uuid]

    def _compile(self, src):
        with warnings.catch_warnings(record=True) as w:
            module = SourceModule(
                src,
                options=['-std=c++14',
                         '-O4',
                         '--use_fast_math',
                         '--expt-relaxed-constexpr',
                         '--maxrregcount=64',
                         '-Xptxas', '-v',
                         '-lineinfo',
                         ] + self.nvcc_extra,
                no_extern_c=True,
                include_dirs=cpp.__path__)
        return module, [str(rec.message) for rec in w]

    @property
    @functools.lru_cache(maxsize=1)
    def template(self):
        return Template(os.path.join(os.path.dirname(__file__), 'template.cu'))

    @property
    def source(self):
        return self._source

    @source.setter
    def source(self, source):
        if self.source != source:
            self._source = source
            self._module = None

    @property
    def module(self):
        if not self._module:
            self._module, self._compiler_message = self._compile(self.source)
        return self._module

    def __call__(self, graphs, node_kernel, edge_kernel, p, q, jobs, starts,
                 output, output_shape, traits, timer):
        ''' transfer graphs and starting probabilities to GPU '''
        timer.tic('transferring graphs to GPU')

        og_last = None
        og_indices = np.empty(len(graphs), np.uint32)
        starting_p = umempty(len(graphs), np.uintp)

        for i, g in enumerate(graphs):
            idx, ps, og = self._register_graph(g, p)
            if i > 0:
                self._assert_homogeneous(og_last, og)
            og_last = og
            og_indices[i] = idx
            starting_p[i] = int(ps.base)

        og_indices_d = umlike(self.graph_cpp[og_indices])

        weighted = og_last.weighted
        node_type = og_last.node_type
        edge_type = og_last.edge_type

        timer.toc('transferring graphs to GPU')

        ''' allocate global job counter '''
        timer.tic('allocate global job counter')
        i_job_global = umzeros(1, np.uint32)
        timer.toc('allocate global job counter')

        ''' code generation '''
        timer.tic('code generation')

        if weighted:
            edge_kernel = TensorProduct(weight=_Multiply(),
                                        label=edge_kernel)

        node_kernel_src = Template(r'''
        using node_theta_t = ${theta_t};

        struct node_kernel_t : node_theta_t {
            template<class V> __device__ __inline__
            auto operator() (V const &v1, V const &v2) const {
                return ${expr};
            }
        };

        __constant__ node_kernel_t node_kernel;
        ''').render(
            theta_t=decltype(node_kernel),
            expr=node_kernel.gen_expr('v1', 'v2')
        )

        edge_kernel_src = Template(r'''
        using edge_theta_t = ${theta_t};

        struct edge_kernel_t : edge_theta_t {
            template<class T> __device__ __inline__
            auto operator() (T const &e1, T const &e2) const {
                return ${expr};
            }
        };

        __constant__ edge_kernel_t edge_kernel;
        ''').render(
            theta_t=decltype(edge_kernel),
            expr=edge_kernel.gen_expr('e1', 'e2')
        )

        self.source = self.template.render(node_kernel=node_kernel_src,
                                           edge_kernel=edge_kernel_src,
                                           node_t=decltype(node_type),
                                           edge_t=decltype(edge_type))
        timer.toc('code generation')

        ''' JIT '''
        timer.tic('JIT')
        kernel = self.module.get_function('graph_kernel_solver')
        timer.toc('JIT')

        ''' calculate launch configuration '''
        timer.tic('calculating launch configuration')
        launch_block_count = (self.device.MULTIPROCESSOR_COUNT
                              * self.block_per_sm)
        shmem_bytes_per_warp = self.module.get_global('shmem_bytes_per_warp')[1]
        shmem_bytes_per_block = (shmem_bytes_per_warp * self.block_size
                                 // self.device.WARP_SIZE)

        max_graph_size = np.max([len(g.nodes) for g in graphs])
        self._allocate_scratch(launch_block_count, max_graph_size**2)

        p_node_kernel, _ = self.module.get_global('node_kernel')
        cuda.memcpy_htod(p_node_kernel, np.array([node_kernel.state],
                                                 dtype=node_kernel.dtype))

        p_edge_kernel, _ = self.module.get_global('edge_kernel')
        cuda.memcpy_htod(p_edge_kernel, np.array([edge_kernel.state],
                                                 dtype=edge_kernel.dtype))

        timer.toc('calculating launch configuration')

        ''' GPU kernel execution '''
        timer.tic('GPU kernel execution')
        kernel(
            og_indices_d,
            starting_p,
            self.scratch_d,
            jobs,
            starts,
            output,
            i_job_global,
            np.uint32(len(jobs)),
            np.uint32(output_shape[0]),
            np.float32(q),
            np.float32(q),  # placeholder for q0
            np.uint32(traits),
            grid=(launch_block_count, 1, 1),
            block=(self.block_size, 1, 1),
            shared=shmem_bytes_per_block,
        )
        self.ctx.synchronize()
        timer.toc('GPU kernel execution')
