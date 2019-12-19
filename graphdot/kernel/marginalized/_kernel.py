#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import numpy as np
from graphdot.util import Timer
from ._backend_factory import backend_factory

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

        def __new__(cls, **kwargs):
            traits = 0
            nodal = kwargs.pop('nodal', False)
            traits |= cls.NODAL if nodal is not False else 0
            traits |= cls.BLOCK if nodal == 'block' else 0
            traits |= cls.SYMMETRIC if kwargs.pop('symmetric', False) else 0
            traits |= cls.LMIN1 if kwargs.pop('lmin1', False) else 0
            traits |= cls.DIAGONAL if kwargs.pop('diagonal', False) else 0
            return traits

    def __init__(self, node_kernel, edge_kernel, p='default', q=0.01,
                 q_bounds=(1e-4, 1 - 1e-4), backend='auto'):
        self.node_kernel = node_kernel
        self.edge_kernel = edge_kernel
        self.p = self._get_starting_probability(p)
        self.q = q
        self.q_bounds = q_bounds

        self.backend = backend_factory(backend)

    def _get_starting_probability(self, p):
        if isinstance(p, str):
            if p == 'uniform' or p == 'default':
                return lambda i, n: 1.0
            else:
                raise ValueError('Unknown starting probability distribution %s'
                                 % self.p)
        else:
            return p

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
        timer = Timer()
        backend = self.backend

        ''' generate jobs '''
        timer.tic('generating jobs')
        if Y is None:
            i, j = np.triu_indices(len(X))
            i, j = i.astype(np.uint32), j.astype(np.uint32)
        else:
            i, j = np.indices((len(X), len(Y)), dtype=np.uint32)
            j += len(X)
        jobs = backend.array(
            np.column_stack((i.ravel(), j.ravel()))
            .ravel()
            .view(np.dtype([('i', np.uint32), ('j', np.uint32)]))
        )
        timer.toc('generating jobs')

        ''' create output buffer '''
        timer.tic('creating output buffer')
        if Y is None:
            starts = backend.zeros(len(X) + 1, dtype=np.uint32)
            if nodal is True:
                sizes = np.array([len(g.nodes) for g in X], dtype=np.uint32)
                np.cumsum(sizes, out=starts[1:])
                n_nodes_X = int(starts[-1])
                output_shape = (n_nodes_X, n_nodes_X)
            else:
                starts[:] = np.arange(len(X) + 1)
                output_shape = (len(X), len(X))
        else:
            starts = backend.zeros(len(X) + len(Y) + 1, dtype=np.uint32)
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
        output = backend.empty(output_shape[0] * output_shape[1], np.float32)
        timer.toc('creating output buffer')

        ''' call GPU kernel '''
        timer.tic('calling GPU kernel (overall)')
        traits = self._Traits(symmetric=Y is None,
                              nodal=nodal,
                              lmin1=lmin == 1)
        backend(np.concatenate((X, Y)) if Y is not None else X,
                self.node_kernel,
                self.edge_kernel,
                self.p,
                self.q,
                jobs,
                starts,
                output,
                output_shape,
                np.uint32(traits),
                timer,
                )
        timer.toc('calling GPU kernel (overall)')

        ''' collect result '''
        timer.tic('collecting result')
        output = output.reshape(*output_shape, order='F')
        timer.toc('collecting result')

        if timer:
            timer.report(unit='ms')
        timer.reset()

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
        timer = Timer()
        backend = self.backend

        ''' generate jobs '''
        timer.tic('generating jobs')
        i = np.arange(len(X), dtype=np.uint32)
        jobs = backend.array(
            np.column_stack((i, i))
            .ravel()
            .view(np.dtype([('i', np.uint32), ('j', np.uint32)]))
        )
        timer.toc('generating jobs')

        ''' create output buffer '''
        timer.tic('creating output buffer')
        starts = backend.zeros(len(X) + 1, dtype=np.uint32)
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
        output = backend.empty(output_length, np.float32)
        timer.toc('creating output buffer')

        ''' call GPU kernel '''
        timer.tic('calling GPU kernel (overall)')
        traits = self._Traits(diagonal=True,
                              nodal=nodal,
                              lmin1=lmin == 1)
        backend(X,
                self.node_kernel,
                self.edge_kernel,
                self.p,
                self.q,
                jobs,
                starts,
                output,
                (output_length, 1),
                traits,
                timer,
                )
        timer.toc('calling GPU kernel (overall)')

        ''' post processing '''
        timer.tic('collecting result')
        if nodal == 'block':
            output = [output[s:s + n**2].reshape(n, n)
                      for s, n in zip(starts[:-1], sizes)]
        timer.toc('collecting result')

        if timer:
            timer.report(unit='ms')
        timer.reset()

        return output

    """⭣⭣⭣⭣⭣ scikit-learn interoperability methods ⭣⭣⭣⭣⭣"""

    def is_stationary(self):
        return False

    @property
    def requires_vector_input(self):
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
        clone = copy.deepcopy(self)
        clone.theta = theta
        return clone

    def get_params(self, deep=False):
        return dict(
            node_kernel=self.node_kernel,
            edge_kernel=self.edge_kernel,
            p=self.p,
            q=self.q,
            q_bounds=self.q_bounds,
            backend=self.backend
        )
