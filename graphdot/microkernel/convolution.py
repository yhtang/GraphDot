#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from graphdot.codegen.cpptool import cpptype
from graphdot.codegen.template import Template
from graphdot.util.pretty_tuple import pretty_tuple
from ._base import MicroKernel


def Convolution(kernel: MicroKernel, mean=True):
    r"""Creates a convolutional microkernel, which averages evaluations of a
    base microkernel between pairs of elements of two variable-length feature
    sequences.

    Parameters
    ----------
    kernel: MicroKernel
        The base kernel can be any elementary or composite microkernels,
        while the attribute to be convolved should be sequences.
    mean: bool
        If True, return the arithmetic mean of the kernel evaluations, i.e.
        :math:`k_{conv}(X, Y) =
        \frac{\sum_{x \in X} \sum_{y \in Y} k_{base}(x, y)}{|X||Y|}`.
        Otherwise, return the sum of the kernel evaluations, i.e.
        :math:`k_{conv}(X, Y) = \sum_{x \in X} \sum_{y \in Y} k_{base}(x, y)`.
        Thus, this serves as a means of normalization beyonds the dot product
        based one.
    """

    @cpptype(kernel=kernel.dtype)
    class ConvolutionOf(MicroKernel):
        @property
        def name(self):
            return 'Convolution'

        def __init__(self, kernel, mean):
            self.kernel = kernel
            self.mean = mean

        def __call__(self, X, Y, jac=False):
            if jac is True:
                Fxy, Jxy = list(zip(*[
                    self.kernel(x, y, jac=True) for x in X for y in Y
                ]))
                if self.mean:
                    return np.mean(Fxy), np.mean(Jxy, axis=0)
                else:
                    return np.sum(Fxy), np.sum(Jxy, axis=0)
            else:
                if self.mean:
                    return np.mean([self.kernel(x, y) for x in X for y in Y])
                else:
                    return np.sum([self.kernel(x, y) for x in X for y in Y])

        def __repr__(self):
            return f'{self.name}({repr(self.kernel)})'

        def gen_expr(self, x, y, theta_scope=''):
            F, J = self.kernel.gen_expr('_1', '_2', theta_scope + 'kernel.')
            f = Template(
                r'''convolution<${mean}>(
                        [&](auto _1, auto _2){return ${f};}, ${x}, ${y}
                )'''
            ).render(
                mean='true' if self.mean else 'false', x=x, y=y, f=F
            )
            template = Template(
                r'''convolution_jacobian<${mean}>(
                        [&](auto _1, auto _2){return ${j};},
                        ${x}, ${y}
                )'''
            )
            mean = 'true' if self.mean else 'false'
            jacobian = [template.render(mean=mean, x=x, y=y, j=j) for j in J]
            return f, jacobian

        @property
        def theta(self):
            return pretty_tuple(
                self.name,
                ['base']
            )(self.kernel.theta)

        @theta.setter
        def theta(self, seq):
            self.kernel.theta = seq[0]

        @property
        def bounds(self):
            return (self.kernel.bounds,)

        @property
        def minmax(self):
            return self.kernel.minmax

    return ConvolutionOf(kernel, mean=mean)
