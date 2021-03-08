#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest
import numpy as np
from graphdot.kernel import KernelOverMetric


class PairwiseDistance:

    def __init__(self, scale):
        self.scale = scale

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
        distance = np.abs(np.subtract.outer(X, Y if Y is not None else X))
        if eval_gradient is True:
            return self.scale * distance, distance.reshape(*distance.shape, 1)
        else:
            return self.scale * distance

    @property
    def hyperparameters(self):
        return (self.scale,)

    @property
    def theta(self):
        return np.log([self.scale])

    @theta.setter
    def theta(self, value):
        self.scale = np.exp(value)[0]

    @property
    def bounds(self):
        return np.log([[1e-4, 1e4]])

    def clone_with_theta(self, theta=None):
        if theta is None:
            theta = self.theta
        clone = type(self)(scale=self.scale)
        clone.theta = theta
        return clone


def test_gauss():

    kernel = KernelOverMetric(
        distance=PairwiseDistance(1.0),
        expr='v * exp(-d^2 / ell^2)',
        x='d',
        v=(1.0, (1e-2, 1e2)),
        ell=(1.0, (1e-2, 1e2))
    )

    X = np.arange(3)
    Y = np.arange(4)

    assert kernel(X).shape == (len(X), len(X))
    assert kernel(Y).shape == (len(Y), len(Y))
    assert kernel(X, X).shape == (len(X), len(X))
    assert kernel(Y, Y).shape == (len(Y), len(Y))
    assert kernel(X, Y).shape == (len(X), len(Y))

    assert kernel(X).diagonal() == pytest.approx(kernel.diag(X))
    assert kernel(Y).diagonal() == pytest.approx(kernel.diag(Y))

    assert len(kernel.theta) == len(kernel.distance.theta) + 2
    assert kernel.bounds.shape == (len(kernel.theta), 2)
    assert len(kernel.hyperparameters) == 3

    kclone = kernel.clone_with_theta()
    assert isinstance(kclone, KernelOverMetric)


@pytest.mark.parametrize('X', [
    np.linspace(-1, 1, 4),
    np.linspace(-1, 1, 40),
    np.linspace(-10, 10, 40),
    np.random.randn(10) * 3.0,
    np.random.rand(10) * 3.0,
])
@pytest.mark.parametrize('kernel', [
    KernelOverMetric(
        distance=PairwiseDistance(1.0),
        expr='v * exp(-d^2 / ell^2)',
        x='d',
        v=(1.0, (1e-2, 1e2)),
        ell=(1.0, (1e-2, 1e2))
    ),
    KernelOverMetric(
        distance=PairwiseDistance(1.0),
        expr='v * exp(-d^2 / ell^2)',
        x='d',
        v=(1.0, (1e-2, 1e2)),
        ell=(2.0, (1e-2, 1e2))
    ),
    KernelOverMetric(
        distance=PairwiseDistance(1.5),
        expr='v * (1 + d**2 / (2 * a * ell**2)) ** -a',
        x='d',
        v=(1.2, (1e-5, 1e5)),
        a=(0.9, (1e-5, 1e5)),
        ell=(1.1, (1e-2, 1e2))
    ),
])
def test_gradient(X, kernel):

    _, grad = kernel(X, eval_gradient=True)

    assert grad.shape == (len(X), len(X), len(kernel.theta))

    delta = 1e-2
    for i, _ in enumerate(kernel.theta):
        h_pos, h_neg = np.exp(kernel.theta), np.exp(kernel.theta)
        h_pos[i] += delta
        h_neg[i] -= delta
        pos = kernel.clone_with_theta(np.log(h_pos))
        neg = kernel.clone_with_theta(np.log(h_neg))
        diff = (pos(X) - neg(X)) / (2 * delta)
        assert np.allclose(grad[:, :, i], diff, rtol=1e-3, atol=1e-3)
