#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import copy


class Normalization:
    r"""Normalizes a kernel using the cosine of angle formula:
    :math:`k_\mathrm{normalized}(x, y) = \frac{k(x, y){\sqrt{k(x, x)k(y, y)}}`.

    Parameters
    ----------
    graph_kernel: object
        The graph kernel to be normalized.
    """
    def __init__(self, graph_kernel):
        self.graph_kernel = graph_kernel

    def __call__(self, X, Y=None, eval_gradient=False, **options):
        """Normalized outcome of
        :py:`self.graph_kernel(X, Y, eval_gradient, **options)`.

        Parameters
        ----------
        Inherits that of the graph kernel object.

        Returns
        -------
        Inherits that of the graph kernel object.
        """
        if eval_gradient is True:
            R, dR = self.graph_kernel(X, Y, eval_gradient=True, **options)
            if Y is None:
                ldiag = rdiag = R.diagonal()
            else:
                ldiag, ldDiag = self.graph_kernel.diag(X, True, **options)
                rdiag, rdDiag = self.graph_kernel.diag(Y, True, **options)
            ldiag_inv = 1 / ldiag
            rdiag_inv = 1 / rdiag
            ldiag_rsqrt = np.sqrt(ldiag_inv)
            rdiag_rsqrt = np.sqrt(rdiag_inv)
            K = ldiag_rsqrt[:, None] * R * rdiag_rsqrt[None, :]
            dK = []
            for i in range(dR.shape[-1]):
                dr = dR[:, :, i]
                if Y is None:
                    lddiag = rddiag = dr.diagonal()
                else:
                    lddiag = ldDiag[:, i]
                    rddiag = rdDiag[:, i]
                dk = (
                    ldiag_rsqrt[:, None] * dr * rdiag_rsqrt[None, :]
                    - 0.5 * ldiag_inv[:, None] * K * rdiag_inv[None, :] * (
                        np.outer(lddiag, rdiag) + np.outer(ldiag, rddiag)
                    )
                )
                dK.append(dk)
            dK = np.stack(dK, axis=2)
            return K, dK
        else:
            R = self.graph_kernel(X, Y, **options)
            if Y is None:
                ldiag = rdiag = R.diagonal()
            else:
                ldiag = self.graph_kernel.diag(X, **options)
                rdiag = self.graph_kernel.diag(Y, **options)
            ldiag_inv = 1 / ldiag
            rdiag_inv = 1 / rdiag
            ldiag_rsqrt = np.sqrt(ldiag_inv)
            rdiag_rsqrt = np.sqrt(rdiag_inv)
            K = ldiag_rsqrt[:, None] * R * rdiag_rsqrt[None, :]
            return K

    def diag(self, X, **options):
        """Normalized outcome of :py:`self.graph_kernel.diag(X, **options)`.

        Parameters
        ----------
        Inherits that of the graph kernel object.

        Returns
        -------
        Inherits that of the graph kernel object.
        """
        return np.ones_like(self.graph_kernel.diag(X, **options))

    """⭣⭣⭣⭣⭣ scikit-learn interoperability methods ⭣⭣⭣⭣⭣"""

    @property
    def hyperparameters(self):
        return self.graph_kernel.hyperparameters

    @property
    def theta(self):
        return self.graph_kernel.theta

    @theta.setter
    def theta(self, value):
        self.graph_kernel.theta = value

    @property
    def hyperparameter_bounds(self):
        return self.graph_kernel.hyperparameter_bounds

    @property
    def bounds(self):
        return self.graph_kernel.bounds

    def clone_with_theta(self, theta):
        clone = copy.deepcopy(self)
        clone.theta = theta
        return clone
