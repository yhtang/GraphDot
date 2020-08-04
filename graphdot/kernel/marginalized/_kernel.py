#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import numbers
import itertools as it
from collections import namedtuple
import numpy as np
from graphdot.graph import Graph
from graphdot.util import Timer
from graphdot.util.iterable import fold_like, flatten
from ._backend_factory import backend_factory
from .starting_probability import StartingProbability, Uniform, Adhoc


class MarginalizedGraphKernel:
    """Implements the random walk-based graph similarity kernel as proposed in:
    Kashima, H., Tsuda, K., & Inokuchi, A. (2003).
    Marginalized kernels between labeled graphs. *In Proceedings of the 20th
    international conference on machine learning (ICML-03)* (pp. 321-328).

    Parameters
    ----------
    node_kernel: microkernel
        A kernelet that computes the similarity between individual nodes
    edge_kernel: microkernel
        A kernelet that computes the similarity between individual edge
    kwargs: optional arguments
        p: positive number (default=1.0) or :py:class:`StartingProbability`
            The starting probability of the random walk on each node. Must be
            either a positive number or a concrete subclass instance of
            :py:class:`StartingProbability`.
        q: float in (0, 1)
            The probability for the random walk to stop during each step.
    """
    trait_t = namedtuple(
        'Traits', 'diagonal, symmetric, nodal, lmin, eval_gradient'
    )

    @classmethod
    def traits(cls, diagonal=False, symmetric=False, nodal=False, lmin=0,
               eval_gradient=False):
        traits = cls.trait_t(
            diagonal, symmetric, nodal, lmin, eval_gradient
        )
        if traits.eval_gradient is True:
            if nodal is not False:
                raise ValueError(
                    'Gradients can only be evaluated with nodal=False'
                )
        return traits

    def __init__(self, node_kernel, edge_kernel, p=1.0, q=0.01,
                 q_bounds=(1e-4, 1 - 1e-4), dtype=np.float, backend='auto'):
        self.node_kernel = node_kernel
        self.edge_kernel = edge_kernel
        self.p = self._get_starting_probability(p)
        self.q = q
        self.q_bounds = q_bounds
        self.element_dtype = dtype

        self.backend = backend_factory(backend)

    def _get_starting_probability(self, p):
        if isinstance(p, StartingProbability):
            return p
        elif isinstance(p, tuple) and len(p) == 2:
            f, expr = p
            if callable(f) and isinstance(expr, str):
                return Adhoc(f, expr)
            else:
                raise ValueError(
                    'An ad hoc starting probability must be specified as an'
                    '(callable, C++ expression) pair.'
                )
        elif isinstance(p, numbers.Number):
            if p > 0:
                return Uniform(p)
            else:
                raise ValueError(f'Starting probability {p} < 0.')
        else:
            raise ValueError(f'Unknown starting probability: {p}')

    def __call__(self, X, Y=None, eval_gradient=False, nodal=False, lmin=0,
                 timing=False):
        """Compute pairwise similarity matrix between graphs

        Parameters
        ----------
        X: list of N graphs
            The graphs must all have same node and edge attributes.
        Y: None or list of M graphs
            The graphs must all have same node and edge attributes.
        eval_gradient: Boolean
            If True, computes the gradient of the kernel matrix with respect
            to hyperparameters and return it alongside the kernel matrix.
        nodal: bool
            If True, return node-wise similarities; otherwise, return graphwise
            similarities.
        lmin: 0 or 1
            Number of steps to skip in each random walk path before similarity
            is computed.
            (lmin + 1) corresponds to the starting value of l in the summation
            of Eq. 1 in Tang & de Jong, 2019 https://doi.org/10.1063/1.5078640
            (or the first unnumbered equation in Section 3.3 of Kashima, Tsuda,
            and Inokuchi, 2003).

        Returns
        -------
        kernel_matrix: ndarray
            if Y is None, return a square matrix containing pairwise
            similarities between the graphs in X; otherwise, returns a matrix
            containing similarities across graphs in X and Y.
        gradient: ndarray
            The gradient of the kernel matrix with respect to kernel
            hyperparameters. Only returned if eval_gradient is True.
        """
        timer = Timer()
        backend = self.backend
        traits = self.traits(
            symmetric=Y is None,
            nodal=nodal,
            lmin=lmin,
            eval_gradient=eval_gradient
        )

        ''' assert graph attributes are compatible with each other '''
        all_graphs = list(it.chain(X, Y)) if Y is not None else X
        pred_or_tuple = Graph.has_unified_types(all_graphs)
        if pred_or_tuple is not True:
            group, first, second = pred_or_tuple
            raise TypeError(
                f'The two graphs have mismatching {group} attributes or '
                'attribute types. If the attributes match in name but differ '
                'in type, try `Graph.unify_datatype` as an automatic fix.\n'
                f'First graph: {first}\n'
                f'Second graph: {second}\n'
            )

        ''' generate jobs '''
        timer.tic('generating jobs')
        if traits.symmetric:
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
        if traits.symmetric:
            starts = backend.zeros(len(X) + 1, dtype=np.uint32)
            if traits.nodal is True:
                sizes = np.array([len(g.nodes) for g in X], dtype=np.uint32)
                np.cumsum(sizes, out=starts[1:])
                n_nodes_X = int(starts[-1])
                output_shape = (n_nodes_X, n_nodes_X)
            else:
                starts[:] = np.arange(len(X) + 1)
                output_shape = (len(X), len(X))
        else:
            starts = backend.zeros(len(X) + len(Y) + 1, dtype=np.uint32)
            if traits.nodal is True:
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
        if traits.eval_gradient is True:
            output_shape = (*output_shape, 1 + self.n_dims)
        output = backend.empty(int(np.prod(output_shape)), np.float32)
        timer.toc('creating output buffer')

        ''' call GPU kernel '''
        timer.tic('calling GPU kernel (overall)')
        backend(
            np.concatenate((X, Y)) if Y is not None else X,
            self.node_kernel,
            self.edge_kernel,
            self.p,
            self.q,
            jobs,
            starts,
            output,
            output_shape,
            traits,
            timer,
        )
        timer.toc('calling GPU kernel (overall)')

        ''' collect result '''
        timer.tic('collecting result')
        output = output.reshape(*output_shape, order='F')
        timer.toc('collecting result')

        if timing:
            timer.report(unit='ms')
        timer.reset()

        if traits.eval_gradient is True:
            return (
                output[:, :, 0].astype(self.element_dtype),
                output[:, :, 1:].astype(self.element_dtype)
            )
        else:
            return output.astype(self.element_dtype)

    def diag(self, X, eval_gradient=False, nodal=False, lmin=0, timing=False):
        """Compute the self-similarities for a list of graphs

        Parameters
        ----------
        X: list of N graphs
            The graphs must all have same node attributes and edge attributes.
        eval_gradient: Boolean
            If True, computes the gradient of the kernel matrix with respect
            to hyperparameters and return it alongside the kernel matrix.
        nodal: bool
            If True, returns a vector containing nodal self similarties; if
            False, returns a vector containing graphs' overall self
            similarities; if 'block', return a list of square matrices which
            forms a block-diagonal matrix, where each diagonal block represents
            the pairwise nodal similarities within a graph.
        lmin: 0 or 1
            Number of steps to skip in each random walk path before similarity
            is computed.
            (lmin + 1) corresponds to the starting value of l in the summation
            of Eq. 1 in Tang & de Jong, 2019 https://doi.org/10.1063/1.5078640
            (or the first unnumbered equation in Section 3.3 of Kashima, Tsuda,
            and Inokuchi, 2003).

        Returns
        -------
        diagonal: numpy.array or list of np.array(s)
            If nodal=True, returns a vector containing nodal self similarties;
            if nodal=False, returns a vector containing graphs' overall self
            similarities; if nodal = 'block', return a list of square matrices,
            each being a pairwise nodal similarity matrix within a graph.
        gradient
            The gradient of the kernel matrix with respect to kernel
            hyperparameters. Only returned if eval_gradient is True.
        """
        timer = Timer()
        backend = self.backend
        traits = self.traits(
            diagonal=True,
            nodal=nodal,
            lmin=lmin,
            eval_gradient=eval_gradient
        )

        ''' assert graph attributes are compatible with each other '''
        pred_or_tuple = Graph.has_unified_types(X)
        if pred_or_tuple is not True:
            group, first, second = pred_or_tuple
            raise TypeError(
                f'The two graphs have mismatching {group} attributes or '
                'attribute types.'
                'If the attribute names do match, then try to unify data '
                'types automatically with `Graph.unify_datatype`.\n'
                f'First graph: {first}\n'
                f'Second graph: {second}\n'
            )

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
        if traits.eval_gradient is True:
            output_shape = (output_length, 1 + self.n_dims)
        else:
            output_shape = (output_length, 1)
        output = backend.empty(int(np.prod(output_shape)), np.float32)
        timer.toc('creating output buffer')

        ''' call GPU kernel '''
        timer.tic('calling GPU kernel (overall)')
        backend(
            X,
            self.node_kernel,
            self.edge_kernel,
            self.p,
            self.q,
            jobs,
            starts,
            output,
            output_shape,
            traits,
            timer,
        )
        timer.toc('calling GPU kernel (overall)')

        ''' post processing '''
        timer.tic('collecting result')
        output = output.reshape(*output_shape, order='F')
        if nodal == 'block':
            retval = [output[s:s + n**2].reshape(n, n)
                      for s, n in zip(starts[:-1], sizes)]
        elif traits.eval_gradient is True:
            retval = (
                output[:, 0].ravel().astype(self.element_dtype),
                output[:, 1:].astype(self.element_dtype)
            )
        else:
            retval = output.ravel().astype(self.element_dtype)
        timer.toc('collecting result')

        if timing:
            timer.report(unit='ms')
        timer.reset()

        return retval

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
    def hyperparameters(self):
        return namedtuple(
            'GraphKernelHyperparameters',
            ['p', 'q', 'node', 'edge']
        )(self.p.theta,
          self.q,
          self.node_kernel.theta,
          self.edge_kernel.theta)

    @property
    def theta(self):
        return np.log(np.fromiter(flatten(self.hyperparameters), np.float))

    @theta.setter
    def theta(self, value):
        (self.p.theta,
         self.q,
         self.node_kernel.theta,
         self.edge_kernel.theta
         ) = fold_like(np.exp(value), self.hyperparameters)

    @property
    def hyperparameter_bounds(self):
        return namedtuple(
            'GraphKernelHyperparameterBounds',
            ['p', 'q', 'node', 'edge']
        )(self.p.bounds,
          self.q_bounds,
          self.node_kernel.bounds,
          self.edge_kernel.bounds)

    @property
    def bounds(self):
        return np.log(np.fromiter(flatten(self.hyperparameter_bounds),
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
