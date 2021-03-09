#!/usr/bin/env python
# -*- coding: utf-8 -*-
from collections import namedtuple
import numpy as np
from graphdot.graph import Graph
from graphdot.util import Timer
from graphdot.kernel.marginalized import MarginalizedGraphKernel
from ._backend_cuda import AltCUDABackend


class AltMarginalizedGraphKernel(MarginalizedGraphKernel):

    trait_t = namedtuple(
        'Traits', 'lmin'
    )

    @classmethod
    def traits(cls, lmin=0):
        traits = cls.trait_t(lmin)
        return traits

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.backend = AltCUDABackend(**kwargs)

    def __call__(self, X, ij, lmin=0, timing=False):
        """Compute a list of pairwise similarities between graphs.

        Parameters
        ----------
        X: list of N graphs
            The graphs must all have same node and edge attributes.
        ij: list of pairs of ints
            Pair indices for which the graph kernel is to be evaluated.
        lmin: 0 or 1
            Number of steps to skip in each random walk path before similarity
            is computed.
            (lmin + 1) corresponds to the starting value of l in the summation
            of Eq. 1 in Tang & de Jong, 2019 https://doi.org/10.1063/1.5078640
            (or the first unnumbered equation in Section 3.3 of Kashima, Tsuda,
            and Inokuchi, 2003).

        Returns
        -------
        gramian: ndarray
            A vector with the same length as ij
        """
        timer = Timer()
        backend = self.backend
        traits = self.traits(
            lmin=lmin,
        )

        ''' assert graph attributes are compatible with each other '''
        pred_or_tuple = Graph.has_unified_types(X)
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
        jobs = backend.array(
            np.array(ij, dtype=np.uint32)
            .ravel()
            .view(dtype=np.dtype([('i', np.uint32), ('j', np.uint32)]))
        )
        timer.toc('generating jobs')

        ''' create output buffer '''
        timer.tic('creating output buffer')
        gramian = backend.empty(len(jobs), np.float32)

        timer.toc('creating output buffer')

        ''' call GPU kernel '''
        timer.tic('calling GPU kernel (overall)')
        backend(
            X,
            self.node_kernel,
            self.edge_kernel,
            self.p,
            self.q,
            self.eps,
            self.ftol,
            self.gtol,
            jobs,
            gramian,
            traits,
            timer,
        )
        timer.toc('calling GPU kernel (overall)')

        ''' collect result '''
        timer.tic('collecting result')
        gramian = gramian.astype(self.element_dtype)
        timer.toc('collecting result')

        if timing:
            timer.report(unit='ms')
        timer.reset()

        return gramian
