#!/usr/bin/env python
# -*- coding: utf-8 -*-
from collections import namedtuple
import numpy as np
from graphdot.codegen.cpptool import cpptype
from graphdot.codegen.template import Template
from ._base import MicroKernel


def Normalize(kernel: MicroKernel):
    r"""Normalize the value range of a microkernel to [0, 1] using the cosine
    of angle formula: :math:`k_{normalized}(x, y) = \frac{k(x, y)}
    {\sqrt{k(x, x) k(y, y)}}`.

    Parameters
    ----------
    kernel:
        The microkernel to be normalized.
    """

    @cpptype(kernel=kernel.dtype)
    class Normalized(MicroKernel):
        @property
        def name(self):
            return 'Normalize'

        def __init__(self, kernel):
            self.kernel = kernel

        def __call__(self, X, Y, jac=False):
            if jac is True:
                Fxx, Jxx = self.kernel(X, X, jac=True)
                Fxy, Jxy = self.kernel(X, Y, jac=True)
                Fyy, Jyy = self.kernel(Y, Y, jac=True)

                if Fxx > 0 and Fyy > 0:
                    return (
                        Fxy * (Fxx * Fyy)**-0.5,
                        (Jxy * (Fxx * Fyy)**-0.5
                         - (0.5 * Fxy * (Fxx * Fyy)**-1.5
                            * (Jxx * Fyy + Fxx * Jyy)))
                    )
                else:
                    return (0.0, np.zeros_like(Jxy))
            else:
                Fxx = self.kernel(X, X)
                Fxy = self.kernel(X, Y)
                Fyy = self.kernel(Y, Y)
                return Fxy * (Fxx * Fyy)**-0.5 if Fxx > 0 and Fyy > 0 else 0.0

        def __repr__(self):
            return f'{self.name}({repr(self.kernel)})'

        def gen_expr(self, x, y, theta_scope=''):
            F, J = self.kernel.gen_expr(
                '_1', '_2', theta_scope + 'kernel.'
            )
            f = Template(
                r'normalize([&](auto _1, auto _2){return ${f};}, ${x}, ${y})'
            ).render(
                x=x, y=y, f=F
            )
            template = Template(
                r'''normalize_jacobian(
                        [&](auto _1, auto _2){return ${f};},
                        [&](auto _1, auto _2){return ${j};},
                        ${x},
                        ${y}
                    )'''
            )
            jacobian = [template.render(x=x, y=y, f=F, j=j) for j in J]
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

    return Normalized(kernel)
