#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import functools
import numpy as np
import pycuda.driver as cuda
from graphdot.codegen import Template
from graphdot.codegen.cpptool import decltype
from graphdot.cuda.array import umempty, umzeros
from graphdot.microkernel import TensorProduct, Product
from graphdot.kernel.marginalized._backend_cuda import CUDABackend
from graphdot.kernel.marginalized._octilegraph import OctileGraph


class MaxiMinBackend(CUDABackend):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    @functools.lru_cache(maxsize=1)
    def template(self):
        return Template(os.path.join(os.path.dirname(__file__), '_backend.cu'))

    def __call__(self, graphs, diags, node_kernel, edge_kernel, p, q, jobs,
                 starts, gramian, gradient, nX, nY, nJ, traits, timer):
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
            'shmem_bytes_per_warp')[1]
        shmem_bytes_per_block = (shmem_bytes_per_warp * self.block_size
                                 // self.device.WARP_SIZE)

        max_graph_size = np.max([len(g.nodes) for g in graphs])
        self._allocate_scratch(launch_block_count, 2 * max_graph_size**2)

        p_node_kernel, _ = self.module.get_global('node_kernel')
        cuda.memcpy_htod(p_node_kernel, np.array([node_kernel.state],
                                                 dtype=node_kernel.dtype))

        p_edge_kernel, _ = self.module.get_global('edge_kernel')
        cuda.memcpy_htod(p_edge_kernel, np.array([edge_kernel.state],
                                                 dtype=edge_kernel.dtype))

        p_p_start, _ = self.module.get_global('p_start')
        cuda.memcpy_htod(p_p_start, np.array([p.state], dtype=p.dtype))

        timer.toc('calculating launch configuration')

        ''' GPU kernel execution '''
        timer.tic('GPU kernel execution')
        kernel(
            graphs_d,
            diags,
            self.scratch_d,
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
            grid=(launch_block_count, 1, 1),
            block=(self.block_size, 1, 1),
            shared=shmem_bytes_per_block,
        )
        self.ctx.synchronize()
        timer.toc('GPU kernel execution')
