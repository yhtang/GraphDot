#!/usr/bin/env python
# -*- coding: utf-8 -*-
from collections import namedtuple
import numpy as np
from graphdot.codegen.cpptool import cpptype
from graphdot.codegen.template import Template
from ._base import MicroKernel
from .normalize import Normalize


def Convolution(kernel: MicroKernel, normalize=True):
    r"""Creates a convolutional microkernel, which sums up evaluations of a
    base microkernel between pairs of elements of two variable-length feature
    sequences.
    :math:`k_{conv}(X, Y) = \sum_{x \in X} \sum_{y \in Y} k_{base}(x, y)`

    Parameters
    ----------
    kernel: MicroKernel
        The base kernel can be any elementary or composite microkernels,
        while the attribute to be convolved should be sequences.
    normalize: bool
        Whether or not to normalize the convolution result to ensure it is a
        valid microkernel in range [0, 1].
    """

    @cpptype(kernel=kernel.dtype)
    class ConvolutionOf(MicroKernel):
        @property
        def name(self):
            return 'Convolution'

        def __init__(self, kernel):
            self.kernel = kernel

        def __call__(self, X, Y, jac=False):
            if jac is True:
                Fxy, Jxy = list(zip(*[
                    self.kernel(x, y, jac=True) for x in X for y in Y
                ]))
                return np.sum(Fxy), np.sum(Jxy, axis=0)
            else:
                return np.sum([self.kernel(x, y) for x in X for y in Y])

        def __repr__(self):
            return f'{self.name}({repr(self.kernel)})'

        def gen_expr(self, x, y, theta_scope=''):
            F, J = self.kernel.gen_expr('_1', '_2', theta_scope + 'kernel.')
            f = Template(
                r'convolution([&](auto _1, auto _2){return ${f};}, ${x}, ${y})'
            ).render(
                x=x, y=y, f=F
            )
            template = Template(
                r'''convolution_jacobian(
                        [&](auto _1, auto _2){return ${j};},
                        ${x}, ${y}
                )'''
            )
            jacobian = [template.render(x=x, y=y, j=j) for j in J]
            return f, jacobian

        @property
        def theta(self):
            return namedtuple(
                f'{self.name}Hyperparameters',
                ['kernel']
            )(self.kernel.theta)

        @theta.setter
        def theta(self, seq):
            self.kernel.theta = seq[0]

        @property
        def bounds(self):
            return (self.kernel.bounds,)

    if normalize is True:
        return Normalize(ConvolutionOf(kernel))
    else:
        return ConvolutionOf(kernel)
