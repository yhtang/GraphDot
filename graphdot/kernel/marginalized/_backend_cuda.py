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
from graphdot.codegen.cpptool import decltype
from graphdot.cuda.array import umempty, umzeros, umarray
from graphdot.microkernel import TensorProduct, Product
from graphdot.util.iterable import flatten, fold_like
from ._backend import Backend
from ._scratch import PCGScratch
from ._octilegraph import OctileGraph


class CUDABackend(Backend):
    """

    Parameters
    ----------
    context: :py:class:`pycuda.driver.Context` instance
        The CUDA context for launching kernels, Will use a default one if
        none is given.
    block_per_sm: int
        Tunes the GPU kernel.
    block_size: int
        Tunes the GPU kernel.
    """

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
        self.uuid = uuid.uuid4()
        self.ctx = kwargs.pop('cuda_context', graphdot.cuda.defctx)
        self.device = self.ctx.get_device()
        self.scratch_pcg = None
        self.scratch_pcg_d = None

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
            assert(x.node_t == y.node_t)
            assert(x.edge_t == y.edge_t)
        except AssertionError as e:
            raise TypeError(
                f'All nodes/edges must be of the same type: {str(e)}'
                'If the graph attributes match in name but differ in type, '
                'try to normalize automatically with `Graph.normalize_types`.'
            )

    def _allocate_scratch(self, scratch, scratch_d, number, length,
                          n_temporaries):
        if (scratch is None or len(scratch) < number or
                scratch[0].nmax < length or
                scratch[0].ndim < n_temporaries):
            self.ctx.synchronize()
            scratch = [
                PCGScratch(length, n_temporaries) for _ in range(number)
            ]
            scratch_d = umarray(
                np.array([s.state for s in scratch], PCGScratch.dtype)
            )
            self.ctx.synchronize()
        return scratch, scratch_d

    def allocate_pcg_scratch(self, number, max_graph_size, traits):
        if traits.eval_gradient is True:
            if traits.nodal in [True, 'block']:
                length = max_graph_size**2
                n_temporaries = 7
            else:
                length = max_graph_size**2 * 2
                n_temporaries = 5
        else:
            length = max_graph_size**2
            n_temporaries = 5

        self.scratch_pcg, self.scratch_pcg_d = self._allocate_scratch(
            self.scratch_pcg, self.scratch_pcg_d, number, length,
            n_temporaries
        )
        return self.scratch_pcg_d

    def _register_graph(self, graph):
        if self.uuid not in graph.cookie:
            # convert to GPU format
            og = OctileGraph(graph)
            graph.cookie[self.uuid] = (og, og.state)
        return graph.cookie[self.uuid]

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
                include_dirs=cpp.__path__,
                # keep=True
            )
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

    @staticmethod
    def gencode_kernel(kernel, name):
        fun, jac = kernel.gen_expr('x1', 'x2')

        return Template(r'''
        using ${name}_theta_t = ${theta_t};

        struct ${name}_t : ${name}_theta_t {

            constexpr static int jac_dims = ${jac_dims};

            template<class X>
            __device__ __inline__
            auto operator() (X const &x1, X const &x2) const {
                return ${expr};
            }

            template<class X>
            __device__ __inline__
            auto _j_a_c_o_b_i_a_n_(X const &x1, X const &x2) const {
                graphdot::array<float, jac_dims> j;
                ${jac;\n};
                return j;
            }
        };

        __constant__ ${name}_t ${name};
        __constant__ ${name}_t ${name}_diff_grid[2 * ${n_theta}];
        __constant__ float32   ${name}_flat_theta[${n_theta}];
        ''').render(
            name=name,
            jac_dims=len(jac),
            theta_t=decltype(kernel),
            expr=fun,
            jac=[f'j[{i}] = {expr}' for i, expr in enumerate(jac)],
            n_theta=len(list(flatten(kernel.theta)))
        )

    @staticmethod
    def gencode_probability(pfunc, name):
        fun, jac = pfunc.gen_expr()

        return Template(r'''
        using ${name}_theta_t = ${theta_t};

        struct ${name}_t : ${name}_theta_t {

            constexpr static int jac_dims = ${jac_dims};

            template<class N>
            __device__ __inline__
            auto operator() (N const &n) const {
                return ${expr};
            }

            template<class N>
            __device__ __inline__
            auto _j_a_c_o_b_i_a_n_(N const &n) const {
                graphdot::array<float, jac_dims> j;
                ${jac;\n};
                return j;
            }
        };

        __constant__ ${name}_t ${name};
        ''').render(
            name=name,
            jac_dims=len(jac),
            theta_t=decltype(pfunc),
            expr=fun,
            jac=[f'j[{i}] = {expr}' for i, expr in enumerate(jac)]
        )

    @staticmethod
    def pack_state(object, diff_grid=False, diff_eps=1e-2):
        def _nudge_theta(object, i, delta):
            o = copy.deepcopy(object)
            t = logtheta.copy()
            t[i] += delta
            o.theta = fold_like(np.exp(t), o.theta)
            return o.state

        pack = [object.state]
        if diff_grid is True:
            logtheta = np.log(list(flatten(object.theta)))
            for i, _ in enumerate(logtheta):
                pack.append(_nudge_theta(object, i, diff_eps))
                pack.append(_nudge_theta(object, i, -diff_eps))
        return pack

    def __call__(self, graphs, node_kernel, edge_kernel, p, q, eps, ftol, gtol,
                 jobs, starts, gramian, gradient, nX, nY, nJ, traits, timer):
        ''' transfer graphs and starting probabilities to GPU '''
        timer.tic('transferring graphs to GPU')

        og_last = None
        graphs_d = umempty(len(graphs), dtype=OctileGraph.dtype)
        for i, g in enumerate(graphs):
            og, ogstate = self._register_graph(g)
            if i > 0:
                self._assert_homogeneous(og_last, og)
            og_last = og
            graphs_d[i] = ogstate

        weighted = og_last.weighted
        node_t = og_last.node_t
        edge_t = og_last.edge_t

        timer.toc('transferring graphs to GPU')

        ''' allocate global job counter '''
        timer.tic('allocate global job counter')
        i_job_global = umzeros(1, np.uint32)
        timer.toc('allocate global job counter')

        ''' code generation '''
        timer.tic('code generation')
        if weighted:
            edge_kernel = TensorProduct(weight=Product(),
                                        label=edge_kernel)

        use_theta_grid = all([
            traits.eval_gradient is True,
            traits.nodal in [True, 'block']
        ])
        node_kernel_src = self.gencode_kernel(node_kernel, 'node_kernel')
        edge_kernel_src = self.gencode_kernel(edge_kernel, 'edge_kernel')
        p_start_src = self.gencode_probability(p, 'p_start')

        with self.template.context(traits=traits) as template:
            self.source = template.render(
                node_kernel=node_kernel_src,
                edge_kernel=edge_kernel_src,
                p_start=p_start_src,
                node_t=decltype(node_t),
                edge_t=decltype(edge_t)
            )
        timer.toc('code generation')

        ''' JIT '''
        timer.tic('JIT')
        kernel = self.module.get_function('graph_kernel_solver')
        timer.toc('JIT')

        ''' calculate launch configuration '''
        timer.tic('calculating launch configuration')
        launch_block_count = (self.device.MULTIPROCESSOR_COUNT
                              * self.block_per_sm)
        shmem_bytes_per_warp = self.module.get_global(
            'shmem_bytes_per_warp'
        )[1]
        shmem_bytes_per_block = (shmem_bytes_per_warp * self.block_size
                                 // self.device.WARP_SIZE)

        ''' allocate scratch buffers '''
        max_graph_size = np.max([len(g.nodes) for g in graphs])
        scratch_pcg = self.allocate_pcg_scratch(
            launch_block_count, max_graph_size, traits
        )

        ''' copy micro kernel parameters to GPU '''
        for name, uker in [('node_kernel', node_kernel),
                           ('edge_kernel', edge_kernel)]:
            states = np.array(
                self.pack_state(uker, diff_grid=use_theta_grid, diff_eps=eps),
                dtype=uker.dtype
            )

            p_uker, _ = self.module.get_global(name)
            cuda.memcpy_htod(p_uker, states[:1])

            if use_theta_grid:
                p_diff_grid, _ = self.module.get_global(f'{name}_diff_grid')
                p_flat_theta, _ = self.module.get_global(f'{name}_flat_theta')
                cuda.memcpy_htod(p_diff_grid, states[1:])
                cuda.memcpy_htod(
                    p_flat_theta,
                    np.fromiter(flatten(uker.theta), dtype=np.float32)
                )

        p_p_start, _ = self.module.get_global('p_start')
        cuda.memcpy_htod(
            p_p_start, np.array([p.state], dtype=p.dtype)
        )

        timer.toc('calculating launch configuration')

        ''' GPU kernel execution '''
        timer.tic('GPU kernel execution')
        kernel(
            graphs_d,
            scratch_pcg,
            jobs,
            starts,
            gramian,
            gradient if gradient is not None else np.uintp(0),
            i_job_global,
            np.uint32(len(jobs)),
            np.uint32(nX),
            np.uint32(nY),
            np.uint32(nJ),
            np.float32(q),
            np.float32(q),  # placeholder for q0
            np.float32(eps),
            np.float32(ftol),
            np.float32(gtol),
            grid=(launch_block_count, 1, 1),
            block=(self.block_size, 1, 1),
            shared=shmem_bytes_per_block,
        )
        self.ctx.synchronize()
        timer.toc('GPU kernel execution')
