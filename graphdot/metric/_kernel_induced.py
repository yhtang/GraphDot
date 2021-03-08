#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np


class KernelInducedDistance:
    r'''The kernel-induced distance is defined by
    :py:math:`d(x, y) = \sqrt{\frac{1}{2}(k(x, x) + k(y, y)) - k(x, y)}`.

    Parameters
    ----------
    kernel: callable
        A positive semidefinite kernel such as one from
        :py:mod:`graphdot.kernel`.
    kernel_options: dict
        Additional arguments to be forwarded to the kernel.
    '''

    def __init__(self, kernel, kernel_options={}):
        self.kernel = kernel
        self.kernel_options = kernel_options

    def __call__(self, X, Y=None, eval_gradient=False):
        '''Computes the distance matrix and optionally its gradient with
        respect to hyperparameters.

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

        Returns
        -------
        distance: 2D matrix
            A distance matrix between the data points.
        gradient: 3D tensor
            A tensor where the i-th frontal slide [:, :, i] contain the partial
            derivative of the distance matrix with respect to the i-th
            hyperparameter. Only returned if the ``eval_gradient`` argument
            is True.
        '''
        if Y is None:
            if eval_gradient is True:
                K12, dK12 = self.kernel(
                    X, eval_gradient=True, **self.kernel_options
                )
                K1 = K2 = K12.diagonal().copy()
                dK1 = dK2 = dK12[np.diag_indices_from(K12)].copy()
            else:
                K12 = self.kernel(X, **self.kernel_options)
                K1 = K2 = K12.diagonal().copy()
        else:
            if eval_gradient is True:
                K12, dK12 = self.kernel(
                    X, Y, eval_gradient=True, **self.kernel_options
                )
                K1, dK1 = self.kernel.diag(
                    X, eval_gradient=True, **self.kernel_options
                )
                K2, dK2 = self.kernel.diag(
                    Y, eval_gradient=True, **self.kernel_options
                )
            else:
                K12 = self.kernel(X, Y, **self.kernel_options)
                K1 = self.kernel.diag(X, **self.kernel_options)
                K2 = self.kernel.diag(Y, **self.kernel_options)

        # some number tweaks to ensure numeric stability
        half = 0.4999997
        eps = 0.0001

        '''More readable, but memory-inefficient version:
        ```
        distance = np.sqrt(
            np.maximum(0.0, -K12 + half * K1[:, None] + half * K2[None, :])
        )
        ```
        '''
        temp1 = np.negative(K12, out=K12)
        temp1 += half * K1[:, None]
        temp1 += half * K2[None, :]
        np.maximum(temp1, 0.0, out=temp1)
        distance = np.sqrt(temp1, out=temp1)

        if eval_gradient is True:
            '''More readable, but memory-inefficient version:
            ```
            gradient = (
                -dK12 + 0.5 * dK1[:, None, :] + 0.5 * dK2[None, :, :]
            ) * (0.5 / (distance + eps))[:, :, None]
            ```
            '''
            temp2 = np.negative(dK12, out=dK12)
            temp2 += 0.5 * dK1[:, None, :]
            temp2 += 0.5 * dK2[None, :, :]
            gradient = np.multiply(
                temp2,
                0.5 / (distance + eps)[:, :, None],
                out=temp2
            )
            return distance, gradient
        else:
            return distance

    @property
    def hyperparameters(self):
        return self.kernel.hyperparameters

    @property
    def theta(self):
        return self.kernel.theta

    @theta.setter
    def theta(self, value):
        self.kernel.theta = value

    @property
    def bounds(self):
        return self.kernel.bounds

    def clone_with_theta(self, theta=None):
        if theta is None:
            theta = self.theta
        return type(self)(
            self.kernel.clone_with_theta(theta),
            self.kernel_options
        )
