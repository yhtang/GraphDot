#!/usr/bin/env python
# -*- coding: utf-8 -*-
import itertools as it
import numpy as np
from graphdot.graph import Graph
from graphdot.kernel.marginalized import MarginalizedGraphKernel
from graphdot.util import Timer
from ._backend_cuda import MaxiMinBackend


class MaxiMin(MarginalizedGraphKernel):
    '''The maximin graph distance is a variant of the Hausdorff distance. Given
    the nodal similarity measure defined on individual nodes by the
    marginalized graph kernel, the maximin distance is the greatest of all the
    kernel-induced distances from a node in one graph to the closest node
    in the other graph. Two graphs are close in the maximin distance if every
    node of either graph is close to some node of the other graph.

    Parameters
    ----------
    args: arguments
        Inherits from
        :py:class:`graphdot.kernel.marginalized.MarginalizedGraphKernel`.
    kwargs: keyword arguments
        Inherits from
        :py:class:`graphdot.kernel.marginalized.MarginalizedGraphKernel`.
    '''

    def __init__(self, *args, **kwargs):
        kwargs['dtype'] = np.float32
        super().__init__(*args, **kwargs)
        self.maximin_backend = MaxiMinBackend()

    def __call__(self, X, Y=None, eval_gradient=False, lmin=0, timing=False):
        '''Computes the distance matrix and optionally its gradient with respect
        to hyperparameters.

        Parameters
        ----------
        X: list of graphs
            The first dataset to be compared.
        Y: list of graphs or None
            The second dataset to be compared. If None, X will be compared with
            itself.
        eval_gradient: bool
            If True, returns the gradient of the weight matrix alongside the
            matrix itself.
        options: keyword arguments
            Additional arguments to be passed to the underlying kernel.

        Returns
        -------
        M: 2D ndarray
            A distance matrix between the data points.
        dM: 3D ndarray
            A tensor where the i-th frontal slide [:, :, i] contain the partial
            derivative of the distance matrix with respect to the i-th
            hyperparameter.
        '''
        timer = Timer()
        backend = self.maximin_backend
        traits = self.traits(
            symmetric=Y is None,
            nodal=False,
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
            output_shape = (len(X), len(X))
            starts = backend.zeros(len(X) + 1, dtype=np.uint32)
            starts[:] = np.arange(len(X) + 1)
            starts_nodal = backend.zeros(len(X) + 1, dtype=np.uint32)
            sizes = np.array([len(g.nodes) for g in X], dtype=np.uint32)
            np.cumsum(sizes, out=starts_nodal[1:])
            diag = backend.array(
                self.diag(X, eval_gradient=False, nodal=True, lmin=lmin)
            )
        else:
            output_shape = (len(X), len(Y))
            XY = np.concatenate((X, Y))
            starts = backend.zeros(len(X) + len(Y) + 1, dtype=np.uint32)
            starts[:len(X)] = np.arange(len(X))
            starts[len(X):] = np.arange(len(Y) + 1)
            starts_nodal = backend.zeros(len(XY) + 1, dtype=np.uint32)
            sizes = np.array([len(g.nodes) for g in XY], dtype=np.uint32)
            np.cumsum(sizes, out=starts_nodal[1:])
            diag = backend.array(
                self.diag(XY, eval_gradient=False, nodal=True, lmin=lmin)
            )

        gramian = backend.empty(int(np.prod(output_shape)), np.float32)
        if traits.eval_gradient is True:
            gradient = backend.empty(
                self.n_dims * int(np.prod(output_shape)), np.float32
            )
        else:
            gradient = None

        diags = [backend.array(diag[b:e])
                 for b, e in zip(starts_nodal[:-1], starts_nodal[1:])]
        diags_d = backend.empty(len(diags), dtype=np.uintp)
        diags_d[:] = [int(d.base) for d in diags]

        timer.toc('creating output buffer')

        ''' call GPU kernel '''
        timer.tic('calling GPU kernel (overall)')
        backend(
            np.concatenate((X, Y)) if Y is not None else X,
            diags_d,
            self.node_kernel,
            self.edge_kernel,
            self.p,
            self.q,
            jobs,
            starts,
            gramian,
            gradient,
            output_shape[0],
            output_shape[1] if len(output_shape) >= 2 else 1,
            self.n_dims,
            traits,
            timer,
        )
        timer.toc('calling GPU kernel (overall)')

        ''' collect result '''
        timer.tic('collecting result')
        gramian = gramian.reshape(*output_shape, order='F')
        if gradient is not None:
            gradient = gradient.reshape(
                (*output_shape, self.n_dims), order='F'
            )
        timer.toc('collecting result')

        if timing:
            timer.report(unit='ms')
        timer.reset()

        if traits.eval_gradient is True:
            raise
            return (
                gramian.astype(self.element_dtype),
                gradient.astype(self.element_dtype)
            )
        else:
            return gramian.astype(self.element_dtype)
