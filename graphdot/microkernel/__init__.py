#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module defines base kernels and composibility rules for creating vertex
and edge kernels for the marginalized graph kernel.
"""
from collections import namedtuple, OrderedDict
import numpy as np
import sympy as sy
from sympy.utilities.lambdify import lambdify
from graphdot.codegen import Template
from graphdot.codegen.cpptool import cpptype
from graphdot.codegen.sympy_printer import cudacxxcode

__all__ = ['MicroKernel',
           'Constant',
           'KroneckerDelta',
           'SquareExponential',
           'RationalQuadratic',
           'Normalize',
           'Compose',
           'TensorProduct',
           'Additive',
           'Convolution',
           ]


@cpptype([])
class _Multiply(MicroKernel):
    """
    Direct product is technically not a valid base kernel
    Only used internally for treating weighted edges
    """

    @property
    def name(self):
        return '_Multiply'

    def __call__(self, x1, x2, jac=False):
        if jac is True:
            return x1 * x2, np.array([])
        else:
            return x1 * x2

    def __repr__(self):
        return f'{self.name}()'

    def gen_expr(self, x, y, jac=False, theta_prefix=''):
        f = f'({x} * {y})'
        if jac is True:
            return f, []
        else:
            return f

    @property
    def theta(self):
        return tuple()

    @theta.setter
    def theta(self, seq):
        pass

    @property
    def bounds(self):
        return tuple()


def Constant(c, c_bounds=None):
    r"""Creates a no-op kernel that returns a constant value (often being 1),
    i.e. :math:`k_\mathrm{c}(\cdot, \cdot) \equiv constant`

    Parameters
    ----------
    c: float > 0
        The constant value.

    Returns
    -------
    MicroKernel
        A kernel instance of corresponding behavior
    """
    if c_bounds is None:
        c_bounds = (c, c)

    @cpptype(c=np.float32)
    class ConstantKernel(MicroKernel):
        @property
        def name(self):
            return 'Constant'

        def __init__(self, c, c_bounds):
            self.c = float(c)
            self.c_bounds = c_bounds
            self._assert_bounds('c', c_bounds)

        def __call__(self, i, j, jac=False):
            if jac is True:
                return self.c, np.ones(1)
            else:
                return self.c

        def __repr__(self):
            return f'{self.name}({self.c})'

        def gen_expr(self, x, y, jac=False, theta_prefix=''):
            f = f'{theta_prefix}c'
            if jac is True:
                return f, ['1.0f']
            else:
                return f

        @property
        def theta(self):
            return namedtuple(
                f'{self.name}Hyperparameters',
                ['c']
            )(self.c)

        @theta.setter
        def theta(self, seq):
            self.c = seq[0]

        @property
        def bounds(self):
            return (self.c_bounds,)

    return ConstantKernel(c, c_bounds)


def KroneckerDelta(h, h_bounds=(1e-3, 1)):
    r"""Creates a Kronecker delta kernel that returns either h or 1 depending
    on whether two objects compare equal, i.e. :math:`k_\delta(i, j) =
    \begin{cases} 1, i = j \\ h, otherwise \end{cases}`

    Parameters
    ----------
    h: float in (0, 1)
        The value of the kernel when two objects do not compare equal

    Returns
    -------
    MicroKernel
        A kernel instance of corresponding behavior
    """

    @cpptype(h=np.float32)
    class KroneckerDeltaKernel(MicroKernel):

        @property
        def name(self):
            return 'KroneckerDelta'

        def __init__(self, h, h_bounds):
            self.h = float(h)
            self.h_bounds = h_bounds
            self._assert_bounds('h', h_bounds)

        def __call__(self, i, j, jac=False):
            if jac is True:
                return (
                    1.0 if i == j else self.h,
                    np.array([0.0 if i == j else 1.0])
                )
            else:
                return 1.0 if i == j else self.h

        def __repr__(self):
            return f'{self.name}({self.h})'

        def gen_expr(self, x, y, jac=False, theta_prefix=''):
            f = f'({x} == {y} ? 1.0f : {theta_prefix}h)'
            if jac is True:
                return f, [f'({x} == {y} ? 0.0f : 1.0f)']
            else:
                return f

        @property
        def theta(self):
            return namedtuple(
                f'{self.name}Hyperparameters',
                ['h']
            )(self.h)

        @theta.setter
        def theta(self, seq):
            self.h = seq[0]

        @property
        def bounds(self):
            return (self.h_bounds,)

    return KroneckerDeltaKernel(h, h_bounds)


SquareExponential = MicroKernel.create(
    'SquareExponential',

    r"""A square exponential kernel smoothly transitions from 1 to
    0 as the distance between two vectors increases from zero to infinity, i.e.
    :math:`k_\mathrm{se}(\mathbf{x}, \mathbf{y}) = \exp(-\frac{1}{2}
    \frac{\lVert \mathbf{x} - \mathbf{y} \rVert^2}{\sigma^2})`""",

    'exp(-0.5 * (x - y)**2 * length_scale**-2)',

    ('x', 'y'),

    ('length_scale', np.float32, 1e-6, np.inf,
     r"""Determines how quickly should the kernel decay to zero. The kernel has
     a value of approx. 0.606 at one length scale, 0.135 at two length
     scales, and 0.011 at three length scales.""")
)

RationalQuadratic = MicroKernel.create(
    'RationalQuadratic',

    r"""A rational quadratic kernel is equivalent to the sum of many square
    exponential kernels with different length scales. The parameter `alpha`
    tunes the relative weights between large and small length scales. When
    alpha approaches infinity, the kernel is identical to the square
    exponential kernel.""",

    '(1 + (x - y)**2 / (2 * alpha * length_scale**2))**(-alpha)',

    ('x', 'y'),

    ('length_scale', np.float32, 1e-6, np.inf,
     r"""The smallest length scale of the square exponential components."""),
    ('alpha', np.float32, 1e-3, np.inf,
     r"""The relative weights of large-scale square exponential components.
     Larger alpha values leads to a faster decay of the weights for larger
     length scales.""")
)


def Normalize(kernel):
    r"""Normalize a positive semidefinite kernel to produce the cosine of angle
    between the input variables.
    :math:`k_{N}(X, Y) = \frac{k_{base}(X, Y)}
    {\sqrt{k_{base}(X, X) k_{base}(Y, Y)}}`

    Parameters
    ----------
    kernel: base kernel
        An elementary or composite base kernel.
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

        def gen_expr(self, x, y, jac=False, theta_prefix=''):
            F, J = self.kernel.gen_expr(
                '_1', '_2', True, theta_prefix + 'kernel.'
            )
            f = Template(
                r'normalize([&](auto _1, auto _2){return ${f};}, ${x}, ${y})'
            ).render(
                x=x, y=y, f=F
            )
            if jac is True:
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
            else:
                return f

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


def Compose(oper, **kw_kernels):
    r"""Creates a composite base kernel via the usage of a reduction operator
    to combine the outputs of multiple scalar kernels on individual features.
    :math:`k_\mathrm{composite}(X, Y; \mathrm{op}) =
    k_{a_1}(X_{a_1}, Y_{a_1})\,\mathrm{op}\,k_{a_2}(X_{a_2}, Y_{a_2})\,
    \mathrm{op}\,\ldots`

    Parameters
    ----------
    oper: str
        A reduction operator. Due to positive definiteness requirements, the
        available options are currently limited to '+', '*'.
    kw_kernels: dict of attribute=kernel pairs
        The kernels can be any base kernels and their compositions as defined
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
    class Composite(MicroKernel):
        @property
        def name(self):
            return 'Compose'

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

        def gen_expr(self, x, y, jac=False, theta_prefix=''):
            F, J = list(
                zip(*[kernel.gen_expr('%s.%s' % (x, key),
                                      '%s.%s' % (y, key),
                                      True,
                                      '%s%s.' % (theta_prefix, key))
                      for key, kernel in self.kw_kernels.items()])
            )
            f = Template('(${F ${opstr} })').render(opstr=self.opstr, F=F)
            if jac is True:
                jacobian = [
                    self.jgen(F, j, i) for i, _ in enumerate(F) for j in J[i]
                ]
                return f, jacobian
            else:
                return f

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
        setattr(Composite, key,
                property(lambda self, key=key: self.kw_kernels[key]))

    return Composite(oper, **oplib[oper], **kw_kernels)


def TensorProduct(**kw_kernels):
    r"""Alias of `Compose('*', **kw_kernels)`.
    :math:`k_\otimes(X, Y) = \prod_{a \in \mathrm{features}} k_a(X_a, Y_a)`
    """
    return Compose('*', **kw_kernels)


def Additive(**kw_kernels):
    r"""Alias of `Compose('+', **kw_kernels)`.
    :math:`k_\oplus(X, Y) = \sum_{a \in \mathrm{features}} k_a(X_a, Y_a)`
    """
    return Compose('+', **kw_kernels)


def Convolution(kernel, normalize=True):
    r"""Creates a convolution kernel, which sums up evaluations of a base
    kernel on pairs of elements from two sequences.
    :math:`k_{CONV}(X, Y) = \sum_{x \in X} \sum_{y \in Y} k_{base}(x, y)`

    Parameters
    ----------
    kernel: base kernel
        The kernel can be any base kernel or a composition of base kernels in
        this module, while the attribute to be convolved should be
        fixed-length sequences.
    normalize: bool
        Whether or not to normalize the convolution to ensure it is a valid
        base kernel in range [0, 1].
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

        def gen_expr(self, x, y, jac=False, theta_prefix=''):
            F, J = self.kernel.gen_expr(
                '_1', '_2', True, theta_prefix + 'kernel.'
            )
            f = Template(
                r'convolution([&](auto _1, auto _2){return ${f};}, ${x}, ${y})'
            ).render(
                x=x, y=y, f=F
            )
            if jac is True:
                template = Template(
                    r'''convolution_jacobian(
                            [&](auto _1, auto _2){return ${j};},
                            ${x},
                            ${y}
                        )'''
                )
                jacobian = [template.render(x=x, y=y, j=j) for j in J]
                return f, jacobian
            else:
                return f

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
