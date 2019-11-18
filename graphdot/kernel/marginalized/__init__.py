#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import uuid
import warnings
import copy
import numpy as np
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
from graphdot import cpp
from graphdot.codegen import Template
from graphdot.codegen.typetool import decltype
import graphdot.cuda
from graphdot.cuda.array import umempty, umzeros, umarray, umlike
from graphdot.cuda.resizable_array import ResizableArray
from graphdot.util import Timer
from ._scratch import BlockScratch
from ._octilegraph import OctileGraph
from .basekernel import TensorProduct, _Multiply

__all__ = ['MarginalizedGraphKernel']


def flatten(iterable):
    for item in iterable:
        if hasattr(item, '__iter__'):
            yield from flatten(item)
        else:
            yield item


def fold_like(flat, example):
    folded = []
    for item in example:
        if hasattr(item, '__iter__'):
            n = len(list(flatten(item)))
            folded.append(fold_like(flat[:n], item))
            flat = flat[n:]
        else:
            folded.append(flat[0])
            flat = flat[1:]
    return tuple(folded)


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
        context: :py:class:`pycuda.driver.Context` instance
            The CUDA context for launching kernels, Will use a default one if
            none is given.
        block_per_sm: int
            Tunes the GPU kernel.
        block_size: int
            Tunes the GPU kernel.
    """
    class _Traits:
        NODAL = 1
        BLOCK = 2
        SYMMETRIC = 4
        LMIN1 = 8
        DIAGONAL = 16

        @classmethod
        def create(cls, **kwargs):
            traits = 0
            nodal = kwargs.pop('nodal', False)
            traits |= cls.NODAL if nodal is not False else 0
            traits |= cls.BLOCK if nodal == 'block' else 0
            traits |= cls.SYMMETRIC if kwargs.pop('symmetric', False) else 0
            traits |= cls.LMIN1 if kwargs.pop('lmin1', False) else 0
            traits |= cls.DIAGONAL if kwargs.pop('diagonal', False) else 0
            return traits

    _template = os.path.join(os.path.dirname(__file__), 'template.cu')

    def __init__(self, node_kernel, edge_kernel, **kwargs):
        self.node_kernel = node_kernel
        self.edge_kernel = edge_kernel
        self.template = Template(self._template)
        self.scratch = None
        self.scratch_capacity = 0
        self.graph_cache = {}
        self.graph_cpp = ResizableArray(OctileGraph.dtype, allocator='numpy')

        self.p = self._get_starting_probability(kwargs.pop('p', 'default'))
        self.q = kwargs.pop('q', 0.01)
        self.q_bounds = kwargs.pop('q_bounds', (1e-4, 1 - 1e-4))

        self.block_per_sm = kwargs.pop('block_per_sm', 8)
        self.block_size = kwargs.pop('block_size', 128)

        self.nvcc_extra = kwargs.pop('nvcc_extra', [])
        self.ctx = kwargs.pop('cuda_context', graphdot.cuda.defctx)
        self.device = self.ctx.get_device()
        self._source = ''
        self._module = None

        self.timer = Timer()

    def _allocate_scratch(self, count, capacity):
        if (self.scratch is None or len(self.scratch) < count or
                self.scratch[0].capacity < capacity):
            self.ctx.synchronize()
            self.scratch = [BlockScratch(capacity) for _ in range(count)]
            self.scratch_d = umarray(np.array([s.state for s in self.scratch],
                                              BlockScratch.dtype))
            self.scratch_capacity = self.scratch[0].capacity
            self.ctx.synchronize()

    def _register_graph(self, graph):
        if not hasattr(graph, 'uuid'):
            graph.uuid = uuid.uuid4()
        if graph.uuid not in self.graph_cache:
            # convert to GPU format
            og = OctileGraph(graph)
            # assign starting probabilities
            p = umarray(np.array([self.p(*r) for r in graph.nodes.iterrows()],
                                 dtype=np.float32))
            i = len(self.graph_cpp)
            self.graph_cache[graph.uuid] = (i, p, og)
            self.graph_cpp.append(og.state)
        return self.graph_cache[graph.uuid]

    def _get_starting_probability(self, p):
        if isinstance(p, str):
            if p == 'uniform' or p == 'default':
                return lambda i, n: 1.0
            else:
                raise ValueError('Unknown starting probability distribution %s'
                                 % self.p)
        else:
            return p

    def _assert_homogeneous(self, x, y):
        try:
            assert(x.weighted == y.weighted)
            assert(x.node_type == y.node_type)
            assert(x.edge_type == y.edge_type)
        except AssertionError as e:
            raise TypeError('All graphs must be of the same type: %s' %
                            str(e))

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

    def _launch_kernel(self, graphs, jobs, starts, output, output_shape,
                       traits):
        ''' transfer graphs and starting probabilities to GPU '''
        self.timer.tic('transferring graphs to GPU')

        og_last = None
        og_indices = np.empty(len(graphs), np.uint32)
        starting_p = umempty(len(graphs), np.uintp)

        for i, g in enumerate(graphs):
            idx, p, og = self._register_graph(g)
            if i > 0:
                self._assert_homogeneous(og_last, og)
            og_last = og
            og_indices[i] = idx
            starting_p[i] = int(p.base)

        og_indices_d = umlike(self.graph_cpp[og_indices])

        weighted = og_last.weighted
        node_type = og_last.node_type
        edge_type = og_last.edge_type

        self.timer.toc('transferring graphs to GPU')

        ''' allocate global job counter '''
        self.timer.tic('allocate global job counter')
        i_job_global = umzeros(1, np.uint32)
        self.timer.toc('allocate global job counter')

        ''' code generation '''
        self.timer.tic('code generation')

        if weighted:
            edge_kernel = TensorProduct(weight=_Multiply(),
                                        label=self.edge_kernel)
        else:
            edge_kernel = self.edge_kernel

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
            theta_t=decltype(self.node_kernel),
            expr=self.node_kernel.gen_expr('v1', 'v2')
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
        self.timer.toc('code generation')

        ''' JIT '''
        self.timer.tic('JIT')
        kernel = self.module.get_function('graph_kernel_solver')
        self.timer.toc('JIT')

        ''' calculate launch configuration '''
        self.timer.tic('calculating launch configuration')
        launch_block_count = (self.device.MULTIPROCESSOR_COUNT
                              * self.block_per_sm)
        shmem_bytes_per_warp = self.module.get_global('shmem_bytes_per_warp')[1]
        shmem_bytes_per_block = (shmem_bytes_per_warp * self.block_size
                                 // self.device.WARP_SIZE)

        max_graph_size = np.max([len(g.nodes) for g in graphs])
        self._allocate_scratch(launch_block_count, max_graph_size**2)

        p_node_kernel, _ = self.module.get_global('node_kernel')
        cuda.memcpy_htod(p_node_kernel, np.array([self.node_kernel.state],
                                                 dtype=self.node_kernel.dtype))

        p_edge_kernel, _ = self.module.get_global('edge_kernel')
        cuda.memcpy_htod(p_edge_kernel, np.array([self.edge_kernel.state],
                                                 dtype=self.edge_kernel.dtype))

        self.timer.toc('calculating launch configuration')

        ''' GPU kernel execution '''
        self.timer.tic('GPU kernel execution')
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
            np.float32(self.q),
            np.float32(self.q),  # placeholder for q0
            np.uint32(traits),
            grid=(launch_block_count, 1, 1),
            block=(self.block_size, 1, 1),
            shared=shmem_bytes_per_block,
        )
        self.ctx.synchronize()
        self.timer.toc('GPU kernel execution')

    def __call__(self, X, Y=None, nodal=False, lmin=0, timer=False):
        """Compute pairwise similarity matrix between graphs

        Parameters
        ----------
        X: list of N graphs
            The graphs must all have same node and edge attributes.
        Y: None or list of M graphs
            The graphs must all have same node and edge attributes.
        nodal: bool
            If True, return node-wise similarities; otherwise, return graphwise
            similarities.
        lmin: 0 or 1
            Number of steps to skip in each random walk path before similarity
            is computed.
            lmin + 1 corresponds to the starting value of l in the summation
            of Eq. 1 in Tang & de Jong, 2019 https://doi.org/10.1063/1.5078640
            (or the first unnumbered equation as in Kashima, Tsuda, and
            Inokuchi, 2003).

        Returns
        -------
        numpy.array
            if Y is None, return a square matrix containing pairwise
            similarities between the graphs in X; otherwise, returns a matrix
            containing similarities across graphs in X and Y.
        """
        ''' generate jobs '''
        self.timer.tic('generating jobs')
        if Y is None:
            i, j = np.triu_indices(len(X))
            i, j = i.astype(np.uint32), j.astype(np.uint32)
        else:
            i, j = np.indices((len(X), len(Y)), dtype=np.uint32)
            j += len(X)
        jobs = umarray(
            np.column_stack((i.ravel(), j.ravel()))
            .ravel()
            .view(np.dtype([('i', np.uint32), ('j', np.uint32)]))
        )
        self.timer.toc('generating jobs')

        ''' create output buffer '''
        self.timer.tic('creating output buffer')
        if Y is None:
            starts = umzeros(len(X) + 1, dtype=np.uint32)
            if nodal is True:
                sizes = np.array([len(g.nodes) for g in X], dtype=np.uint32)
                np.cumsum(sizes, out=starts[1:])
                n_nodes_X = int(starts[-1])
                output_shape = (n_nodes_X, n_nodes_X)
            else:
                starts[:] = np.arange(len(X) + 1)
                output_shape = (len(X), len(X))
        else:
            starts = umzeros(len(X) + len(Y) + 1, dtype=np.uint32)
            if nodal is True:
                sizes = np.array([len(g.nodes) for g in X]
                                 + [len(g.nodes) for g in Y],
                                 dtype=np.uint32)
                np.cumsum(sizes, out=starts[1:])
                n_nodes_X = int(starts[len(X)])
                starts[len(X):] -= n_nodes_X
                n_nodes_Y = int(starts[-1])
                output_shape = (n_nodes_X, n_nodes_Y)
            else:
                starts[:len(X)] = np.arange(len(X))
                starts[len(X):] = np.arange(len(Y) + 1)
                output_shape = (len(X), len(Y))
        output = umempty(output_shape[0] * output_shape[1], np.float32)
        self.timer.toc('creating output buffer')

        ''' call GPU kernel '''
        self.timer.tic('calling GPU kernel (overall)')
        traits = self._Traits.create(symmetric=Y is None,
                                     nodal=nodal,
                                     lmin1=lmin == 1)
        self._launch_kernel(np.concatenate((X, Y)) if Y is not None else X,
                            jobs,
                            starts,
                            output,
                            output_shape,
                            np.uint32(traits))
        self.ctx.synchronize()
        self.timer.toc('calling GPU kernel (overall)')

        ''' collect result '''
        self.timer.tic('collecting result')
        output = output.reshape(*output_shape, order='F')
        self.timer.toc('collecting result')

        if timer:
            self.timer.report(unit='ms')
        self.timer.reset()

        return output

    def diag(self, X, nodal=False, lmin=0, timer=False):
        """Compute the self-similarities for a list of graphs

        Parameters
        ----------
        X: list of N graphs
            The graphs must all have same node attributes and edge attributes.
        nodal: bool
            If True, returns a vector containing nodal self similarties; if
            False, returns a vector containing graphs' overall self
            similarities; if 'block', return a list of square matrices which
            forms a block-diagonal matrix, where each diagonal block represents
            the pairwise nodal similarities within a graph.
        lmin: 0 or 1
            Number of steps to skip in each random walk path before similarity
            is computed.
            lmin + 1 corresponds to the starting value of l in the summation
            of Eq. 1 in Tang & de Jong, 2019 https://doi.org/10.1063/1.5078640
            (or the first unnumbered equation as in Kashima, Tsuda, and
            Inokuchi, 2003).

        Returns
        -------
        numpy.array or list of np.array(s)
            If nodal=True, returns a vector containing nodal self similarties;
            if nodal=False, returns a vector containing graphs' overall self
            similarities; if nodal = 'block', return a list of square matrices,
            each being a pairwise nodal similarity matrix within a graph.
        """

        ''' generate jobs '''
        self.timer.tic('generating jobs')
        i = np.arange(len(X), dtype=np.uint32)
        jobs = umarray(
            np.column_stack((i, i))
            .ravel()
            .view(np.dtype([('i', np.uint32), ('j', np.uint32)]))
        )
        self.timer.toc('generating jobs')

        ''' create output buffer '''
        self.timer.tic('creating output buffer')
        starts = umzeros(len(X) + 1, dtype=np.uint32)
        if nodal is True:
            sizes = np.array([len(g.nodes) for g in X], dtype=np.uint32)
            np.cumsum(sizes, out=starts[1:])
            output_length = int(starts[-1])
        elif nodal is False:
            starts[:] = np.arange(len(X) + 1)
            output_length = len(X)
        elif nodal == 'block':
            sizes = np.array([len(g.nodes) for g in X], dtype=np.uint32)
            np.cumsum(sizes**2, out=starts[1:])
            output_length = int(starts[-1])
        else:
            raise(ValueError("Invalid 'nodal' option '%s'" % nodal))
        output = umempty(output_length, np.float32)
        self.timer.toc('creating output buffer')

        ''' call GPU kernel '''
        self.timer.tic('calling GPU kernel (overall)')
        traits = self._Traits.create(diagonal=True,
                                     nodal=nodal,
                                     lmin1=lmin == 1)
        self._launch_kernel(X,
                            jobs,
                            starts,
                            output,
                            (output_length, 1),
                            traits)
        self.ctx.synchronize()
        self.timer.toc('calling GPU kernel (overall)')

        ''' post processing '''
        self.timer.tic('collecting result')
        if nodal == 'block':
            output = [output[s:s + n**2].reshape(n, n)
                      for s, n in zip(starts[:-1], sizes)]
        self.timer.toc('collecting result')

        if timer:
            self.timer.report(unit='ms')
        self.timer.reset()

        return output

    """scikit-learn interoperability methods"""

    def is_stationary(self):
        return False

    @property
    def n_dims(self):
        '''p.theta + q + node_kernel.theta + edge_kernel.theta'''
        return len(self.theta)

    @property
    def theta_folded(self):
        return [self.q,
                self.node_kernel.theta,
                self.edge_kernel.theta
                ]

    @property
    def theta(self):
        return np.log(np.fromiter(flatten(self.theta_folded), np.float))

    @theta.setter
    def theta(self, value):
        (self.q,
         self.node_kernel.theta,
         self.edge_kernel.theta
         ) = fold_like(np.exp(value), self.theta_folded)

    @property
    def bounds_folded(self):
        return (self.q_bounds,
                self.node_kernel.bounds,
                self.edge_kernel.bounds)

    @property
    def bounds(self):
        return np.log(np.fromiter(flatten(self.bounds_folded),
                                  np.float)).reshape(-1, 2, order='C')

    def clone_with_theta(self, theta):
        cloned = copy.copy(self)
        cloned.node_kernel = copy.deepcopy(self.node_kernel)
        cloned.edge_kernel = copy.deepcopy(self.edge_kernel)
        cloned.theta = theta
        return cloned
