#!/usr/bin/env python
# -*- coding: utf-8 -*-
from collections import namedtuple
import numpy as np
from graphdot.codegen.cpptool import cpptype
from graphdot.codegen.template import Template
from ._base import MicroKernel


def Composite(oper, **kw_kernels):
    r"""Creates a microkernel on multiple features, which uses a reduction
    operator to combine the outputs of multiple microkernels on individual
    features. :math:`k_\mathrm{composite}(X, Y; \mathrm{op}) =
    k_{a_1}(X_{a_1}, Y_{a_1})\,\mathrm{op}\,k_{a_2}(X_{a_2}, Y_{a_2})\,
    \mathrm{op}\,\ldots`

    Parameters
    ----------
    oper: str
        A reduction operator. Due to positive definiteness requirements, the
        available options are currently limited to '+', '*'.
    kw_kernels: dict of attribute=kernel pairs
        The kernels can be any microkernels and their compositions as defined
        in this module, while features should be strings that represent
        valid Python/C++ identifiers.
    """
    oplib = {
        '+': dict(
            ufunc=np.add,                           # associated numpy ufunc
            jfunc=lambda F, f, j: j,                # Jacobian evaluator
            jgen=lambda F_expr, j_expr, i: j_expr,  # Jacobian code generator
        ),
        '*': dict(
            ufunc=np.multiply,
            jfunc=lambda F, f, j: F / f * j,
            jgen=lambda F_expr, j_expr, i: Template('(${X * })').render(
                X=F_expr[:i] + (j_expr,) + F_expr[i + 1:]
            )
        ),
    }

    if oper not in oplib:
        raise ValueError(f'Invalid reduction operator {repr(oper)}.')

    @cpptype([(key, ker.dtype) for key, ker in kw_kernels.items()])
    class CompositeKernel(MicroKernel):
        @property
        def name(self):
            return 'Composite'

        def __init__(self, opstr, ufunc, jfunc, jgen, **kw_kernels):
            self.opstr = opstr
            self.ufunc = ufunc
            self.jfunc = jfunc
            self.jgen = jgen
            self.kw_kernels = kw_kernels

        def __repr__(self):
            return Template('${cls}(${opstr}, ${kwexpr, })').render(
                cls=self.name,
                opstr=repr(self.opstr),
                kwexpr=[f'{k}={repr(K)}' for k, K in self.kw_kernels.items()])

        def __call__(self, X, Y, jac=False):
            if jac is True:
                F, J = list(
                    zip(*[kernel(X[key], Y[key], True)
                          for key, kernel in self.kw_kernels.items()])
                )
                S = self.ufunc.reduce(F)
                jacobian = np.array([
                    self.jfunc(S, f, j) for i, f in enumerate(F) for j in J[i]
                ])
                return S, jacobian
            else:
                return self.ufunc.reduce([
                    f(X[k], Y[k]) for k, f in self.kw_kernels.items()
                ])

        def gen_expr(self, x, y, theta_scope=''):
            F, J = list(
                zip(*[kernel.gen_expr('%s.%s' % (x, key),
                                      '%s.%s' % (y, key),
                                      '%s%s.' % (theta_scope, key))
                      for key, kernel in self.kw_kernels.items()])
            )
            f = Template('(${F ${opstr} })').render(opstr=self.opstr, F=F)
            jacobian = [
                self.jgen(F, j, i) for i, _ in enumerate(F) for j in J[i]
            ]
            return f, jacobian

        @property
        def theta(self):
            return namedtuple(
                f'{self.name}Hyperparameters',
                self.kw_kernels.keys()
            )(*[k.theta for k in self.kw_kernels.values()])

        @theta.setter
        def theta(self, seq):
            for kernel, value in zip(self.kw_kernels.values(), seq):
                kernel.theta = value

        @property
        def bounds(self):
            return tuple(k.bounds for k in self.kw_kernels.values())

    # for the .state property of cpptype
    for key in kw_kernels:
        setattr(CompositeKernel, key,
                property(lambda self, key=key: self.kw_kernels[key]))

    return CompositeKernel(oper, **oplib[oper], **kw_kernels)
