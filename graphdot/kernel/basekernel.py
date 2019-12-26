#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module defines base kernels and composibility rules for creating vertex
and edge kernels for the marginalized graph kernel.
"""
import re
from collections import namedtuple, OrderedDict
from functools import lru_cache
import numpy as np
import sympy as sy
from sympy.utilities.autowrap import ufuncify
from graphdot.codegen import Template
from graphdot.codegen.typetool import cpptype
from graphdot.codegen.sympy import cuda_cxx11_code_printer

__all__ = ['Kernel',
           'Constant',
           'KroneckerDelta',
           'SquareExponential',
           'TensorProduct']


class Kernel:
    """
    Parent class for all base kernels
    """

    def __add__(self, k):
        r"""Implements the additive kernel composition semantics, i.e.
        expression ``k1 + k2`` creates
        :math:`k_+(a, b) = k_1(a, b) + k_2(a, b)`"""
        return Kernel._op(self, k if isinstance(k, Kernel) else Constant(k),
                          lambda x, y: x + y, '+')

    def __radd__(self, k):
        return Kernel._op(k if isinstance(k, Kernel) else Constant(k), self,
                          lambda x, y: x + y, '+')

    def __mul__(self, k):
        r"""Implements the multiplicative kernel composition semantics, i.e.
        expression ``k1 * k2`` creates
        :math:`k_\times(a, b) = k_1(a, b) \times k_2(a, b)`"""
        return Kernel._op(self, k if isinstance(k, Kernel) else Constant(k),
                          lambda x, y: x * y, '*')

    def __rmul__(self, k):
        return Kernel._op(k if isinstance(k, Kernel) else Constant(k), self,
                          lambda x, y: x * y, '*')

    @staticmethod
    def _op(k1, k2, op, opstr):
        # only works with python >= 3.6
        # @cpptype(k1=k1.dtype, k2=k2.dtype)
        @cpptype([('k1', k1.dtype), ('k2', k2.dtype)])
        class KernelOperator(Kernel):
            def __init__(self, k1, k2):
                self.k1 = k1
                self.k2 = k2

            def __call__(self, i, j):
                return op(self.k1(i, j), self.k2(i, j))

            def __str__(self):
                return '({} {} {})'.format(str(self.k1), opstr, str(self.k2))

            def __repr__(self):
                return '{k1} {o} {k2}'.format(
                    k1=repr(k1),
                    o=opstr,
                    k2=repr(k2))

            def gen_constexpr(self, x, y):
                return '({k1} {op} {k2})'.format(k1=self.k1.gen_constexpr(x, y),
                                                 k2=self.k2.gen_constexpr(x, y),
                                                 op=opstr)

            def gen_expr(self, x, y, theta_prefix=''):
                return '({k1} {op} {k2})'.format(
                    k1=self.k1.gen_expr(x, y, theta_prefix + 'k1.'),
                    k2=self.k2.gen_expr(x, y, theta_prefix + 'k2.'),
                    op=opstr)

            @property
            def theta(self):
                return (self.k1.theta, self.k2.theta)

            @theta.setter
            def theta(self, seq):
                self.k1.theta = seq[0]
                self.k2.theta = seq[1]

            @property
            def bounds(self):
                return tuple(self.k1.bounds, self.k2.bounds)

        return KernelOperator(k1, k2)


def Constant(c, c_bounds=(0, np.inf)):
    r"""Creates a no-op kernel that returns a constant value (often being 1),
    i.e. :math:`k_\mathrm{c}(\cdot, \cdot) \equiv constant`

    Parameters
    ----------
    constant: float > 0
        The value of the kernel

    Returns
    -------
    Kernel
        A kernel instance of corresponding behavior
    """

    # only works with python >= 3.6
    # @cpptype(constant=np.float32)
    @cpptype([('c', np.float32)])
    class ConstantKernel(Kernel):
        def __init__(self, c, c_bounds):
            self.c = float(c)
            self.c_bounds = c_bounds

        def __call__(self, i, j):
            return self.c

        def __str__(self):
            return '{}'.format(self.c)

        def __repr__(self):
            return 'Constant({})'.format(self.c)

        def gen_constexpr(self, x, y):
            return '{:f}f'.format(self.c)

        def gen_expr(self, x, y, theta_prefix=''):
            return '{}c'.format(theta_prefix)

        @property
        def theta(self):
            return (self.c,)

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
    Kernel
        A kernel instance of corresponding behavior
    """

    # only works with python >= 3.6
    # @cpptype(lo=np.float32, hi=np.float32)
    @cpptype([('h', np.float32)])
    class KroneckerDeltaKernel(Kernel):

        def __init__(self, h, h_bounds):
            self.h = float(h)
            self.h_bounds = h_bounds

        def __call__(self, i, j):
            return 1.0 if i == j else self.h

        def __str__(self):
            return 'δ({})'.format(self.h)

        def __repr__(self):
            return 'KroneckerDelta({})'.format(self.h)

        def gen_constexpr(self, x, y):
            return '({} == {} ? 1.0f : {:f}f)'.format(x, y, self.h)

        def gen_expr(self, x, y, theta_prefix=''):
            return '({} == {} ? 1.0f : {p}h)'.format(x, y, p=theta_prefix)

        @property
        def theta(self):
            return (self.h,)

        @theta.setter
        def theta(self, seq):
            self.h = seq[0]

        @property
        def bounds(self):
            return (self.h_bounds,)

    return KroneckerDeltaKernel(h, h_bounds)


# def SquareExponential(length_scale, length_scale_bounds=(1e-6, np.inf)):
#     r"""Creates a square exponential kernel that smoothly transitions from 1 to
#     0 as the distance between two vectors increases from zero to infinity, i.e.
#     :math:`k_\mathrm{se}(\mathbf{x}_1, \mathbf{x}_2) = \exp(-\frac{1}{2}
#     \frac{\lVert \mathbf{x}_1 - \mathbf{x}_2 \rVert^2}{\sigma^2})`

#     Parameters
#     ----------
#     length_scale: float > 0
#         Determines how quickly should the kernel decay to zero. The kernel has
#         a value of approx. 0.606 at one length scale, 0.135 at two length
#         scales, and 0.011 at three length scales.

#     Returns
#     -------
#     Kernel
#         A kernel instance of corresponding behavior
#     """

#     # only works with python >= 3.6
#     # @cpptype(nrsql=np.float32)
#     @cpptype([('nrsql', np.float32)])
#     class SquareExponentialKernel(Kernel):
#         def __init__(self, length_scale, length_scale_bounds):
#             self.length_scale = length_scale
#             self.length_scale_bounds = length_scale_bounds

#         def __call__(self, x1, x2):
#             return np.exp(-0.5 * np.sum((x1 - x2)**2) / self.length_scale**2)

#         def __str__(self):
#             return 'SqExp({})'.format(self.length_scale)

#         def __repr__(self):
#             return 'SquareExponential({})'.format(self.length_scale)

#         def gen_constexpr(self, x, y):
#             return 'expf({:f}f * ({} - {}) * ({} - {}))'.format(
#                 self.nrsql, x, y, x, y)

#         def gen_expr(self, x, y, theta_prefix=''):
#             return 'expf({p}nrsql * ({x} - {y}) * ({x} - {y}))'.format(
#                 p=theta_prefix, x=x, y=y)

#         @property
#         def nrsql(self):
#             return -0.5 / self.length_scale**2

#         @property
#         def theta(self):
#             return (self.length_scale,)

#         @theta.setter
#         def theta(self, seq):
#             self.length_scale = seq[0]

#         @property
#         def bounds(self):
#             return (self.length_scale_bounds,)

#     return SquareExponentialKernel(length_scale, length_scale_bounds)


def make_hyperparameters(kernel, *specifications):

    defs = OrderedDict()

    for spec in specifications:
        if len(spec) == 2:
            symbol, dtype = spec
            defs[symbol] = dict(dtype=dtype)
        if len(spec) == 3:
            symbol, dtype, doc = spec
            defs[symbol] = dict(dtype=dtype, doc=doc)
        elif len(spec) == 4:
            symbol, dtype, lb, ub = spec
            defs[symbol] = dict(dtype=dtype, bounds=(lb, ub))
        elif len(spec) == 5:
            symbol, dtype, doc, lb, ub = spec
            defs[symbol] = dict(dtype=dtype, doc=doc, bounds=(lb, ub))
        else:
            raise ValueError(
                'Invalid hyperparameter specification, '
                'must be one of\n'
                '(symbol, dtype)\n',
                '(symbol, dtype, doc)\n',
                '(symbol, dtype, lb, ub)\n',
                '(symbol, dtype, doc, lb, ub)\n',
            )

    class Meta(type):

        _definitions = defs

        @property
        def definitions(self):
            return self._definitions

        @property
        def names(self):
            return self.definitions.keys()

        @property
        def dtypes(self):
            return [t['dtype'] for t in self.definitions.values()]

    class Hyperparameters(metaclass=Meta):

        _kernel = kernel
        values_t = namedtuple("HyperparameterValues", defs)
        bounds_t = namedtuple("HyperparameterBounds", defs)

        def __init__(self, *args, **kwargs):

            values = {}
            bounds = {}

            for symbol, value in zip(self.names, args):
                values[symbol] = value

            for symbol in self.names:
                try:
                    values[symbol] = kwargs[symbol]
                except KeyError:
                    if symbol not in values:
                        raise KeyError(
                            'Hyperparameter {} not provided for {}'.format(
                                symbol,
                                self._kernel
                            )
                        )

                try:
                    bounds[symbol] = kwargs['%s_bounds' % symbol]
                except KeyError:
                    try:
                        bounds[symbol] = self.definitions[symbol]['bounds']
                    except KeyError:
                        raise KeyError(
                            'Bounds for hyperparameter {} of kernel {} not '
                            'specified, while no default value exists.'.format(
                                symbol,
                                self._kernel
                            )
                        )

            self._values = self.values_t(**values)
            self._bounds = self.bounds_t(**bounds)

        @property
        def definitions(self):
            return Hyperparameters.definitions

        @property
        def names(self):
            return Hyperparameters.names

        @property
        def values(self):
            return self._values

        @values.setter
        def values(self, kwrhs):
            self._values = self.values_t(**kwrhs)

        @property
        def bounds(self):
            return self._bounds

        @bounds.setter
        def bounds(self, kwrhs):
            self._bounds = self.bounds_t(**kwrhs)

    return Hyperparameters


@lru_cache(128)
def uf(v, e):
    return ufuncify(v, e)

def make_kernel(name, expression, variables, *hyperparameter_specs):

    '''parse expression'''
    if isinstance(expression, str):
        expression = sy.sympify(expression)
    from sympy.codegen import rewriting
    opt = rewriting.create_expand_pow_optimization(3)
    expression = opt(expression)

    '''check input variables'''
    if len(variables) != 2:
        raise ValueError('A kernel must have exactly two variables')
    variables = [sy.Symbol(v) if isinstance(v, str) else v for v in variables]

    '''parse the list of hyperparameters'''
    hparams_t = make_hyperparameters(name, *hyperparameter_specs)

    '''create kernel class'''
    @cpptype(list(zip(hparams_t.names, hparams_t.dtypes)))
    class BaseKernel(Kernel):

        __name__ = name

        _expression = expression
        _variables = variables
        _hparams_t = hparams_t

        def __init__(self, *args, **kwargs):
            self._hparams = self._hparams_t(*args, **kwargs)

        @property
        # @cached_on('theta')
        def ufunc(self):
            expr = self._expression.subs(zip(self.theta_names,
                                             self.theta))
            # return ufuncify(self._variables, expr)
            return uf(self._variables, expr)

        # @cached_on('theta')
        def expanded_expr(self):
            return self._expression.subs(zip(self.theta_names,
                                             self.theta))

        def __call__(self, x1, x2):
            return self.ufunc.outer(x1, x2)

        # def __str__(self):
        #     return '%s({})'.format(self.length_scale)

        def __repr__(self):
            return Template('${cls}(${theta, }, ${bounds, })').render(
                cls=self.__name__,
                theta=['{}={}'.format(n, v)
                       for n, v in zip(self.theta_names,
                                       self.theta)],
                bounds=['{}_bounds={}'.format(n, v)
                        for n, v in zip(self.theta_names,
                                        self._hparams.bounds)]
            )

        def gen_constexpr(self, x, y):
            expr = self._expression.subs(zip(self._variables, (x, y)))
            expr = self._expression.subs(zip(self.theta_names, self.theta))
            return cuda_cxx11_code_printer(expr)

        def gen_expr(self, x, y, theta_prefix=''):
            expr = self._expression.subs(zip(self._variables, (x, y)))
            # protect hyperparameters for scoping touchup
            expr = self._expression.subs(
                [(n, '__%s__' % n) for n in self.theta_names]
            )
            expr = cuda_cxx11_code_printer(expr)
            # prefix hyperparameters with correct scope
            for theta in self.theta_names:
                expr = re.sub('__%s__' % theta, theta_prefix + theta, expr)
            return expr

        @property
        def theta(self):
            return self._hparams.values

        @theta.setter
        def theta(self, seq):
            assert(len(seq) == len(self._hparams))
            self._hparams.values = {
                name: value for name, value in zip(self.theta_names, seq)
            }

        @property
        def bounds(self):
            return self._hparams.bounds

        @property
        def theta_names(self):
            return self._hparams.names

    return BaseKernel


SquareExponential = make_kernel(
    'SquareExponential',
    'exp(-(x - y)**2 / (2 * length_scale**2))',
    ('x', 'y'),
    ('length_scale', np.float32, 1e-6, np.inf)
)

# if __name__ == '__main__':

#     SquareExponentialKernel = make_kernel(
#         'SquareExponential',
#         'exp(-(x - y)**2 / (2 * length_scale**2))',
#         ('x', 'y'),
#         ('length_scale', np.float32, 1e-6, np.inf)
#     )

#     print(SquareExponentialKernel)

#     k = SquareExponentialKernel(length_scale=1.0)

#     print(k._expression)
#     # print(k.length_scale)
#     # print(k.dtype)
#     print(repr(k))
#     print(k.theta)
#     print(k.bounds)
#     print(k.gen_constexpr('X', 'Y'))
#     print(k.gen_expr('X', 'Y', 'some.scope.'))
#     print(k(0, 0))
#     print(k(1, 1))
#     print(k(0, 1))


@cpptype([])
class _Multiply(Kernel):
    """
    Direct product is technically not a valid base kernel
    Only used internally for treating weighted edges
    """

    def __call__(self, x1, x2):
        return x1 * x2

    def __str__(self):
        return '_mul'

    def __repr__(self):
        return '_Multiply()'

    def gen_constexpr(self, x, y):
        return '({} * {})'.format(x, y)

    def gen_expr(self, x, y, theta_prefix=''):
        return '({} * {})'.format(x, y)

    @property
    def theta(self):
        return tuple()

    @theta.setter
    def theta(self, seq):
        pass

    @property
    def bounds(self):
        return tuple()

# def Product(*kernels):
#     @cpptype([('k%d' % i, ker.dtype) for i, ker in enumerate(kernels)])
#     class ProductKernel(Kernel):
#         def __init__(self, *kernels):
#             self.kernels = kernels
#
#         def __call__(self, object1, object2):
#             prod = 1.0
#             for kernel in self.kernels:
#                 prod *= kernel(object1, object2)
#             return prod
#
#         def __repr__(self):
#             return ' * '.join([repr(k) for k in self.kernels])
#
#         def gen_constexpr(self, x, y):
#             return ' * '.join([k.gen_constexpr(x, y) for k in self.kernels])
#
#         @property
#         def theta(self):
#             return [k.theta for k in self.kw_kernels.values()]
#
#     return ProductKernel(*kernels)


def TensorProduct(**kw_kernels):
    r"""Creates a tensor product kernel, i.e. a product of multiple kernels
    where each kernel is associated with a keyword-indexed 'attribute'.
    :math:`k_\otimes(X, Y) = \prod_{a \in \mathrm{attributes}} k_a(X_a, Y_a)`

    Parameters
    ----------
    kwargs: dict of attribute=kernel pairs
        The kernels can be any base kernels and their compositions as defined
        in this module, while attributes should be strings that represent
        valid Python/C++ identifiers.
    """

    @cpptype([(key, ker.dtype) for key, ker in kw_kernels.items()])
    class TensorProductKernel(Kernel):
        def __init__(self, **kw_kernels):
            self.kw_kernels = kw_kernels
            # for the .state property of cpptype
            for key in kw_kernels:
                setattr(TensorProductKernel, key,
                        property(lambda self, key=key: self.kw_kernels[key]))

        def __call__(self, object1, object2):
            prod = 1.0
            for key, kernel in self.kw_kernels.items():
                prod *= kernel(object1[key], object2[key])
            return prod

        def __str__(self):
            return ' ⊗ '.join([kw + ':' + str(k)
                               for kw, k in self.kw_kernels.items()])

        def __repr__(self):
            return Template('TensorProduct(${kwexpr, })').render(
                kwexpr=['{}={}'.format(kw, repr(k))
                        for kw, k in self.kw_kernels.items()])

        def gen_constexpr(self, x, y):
            return Template('(${expr*})').render(
                expr=[k.gen_constexpr('%s.%s' % (x, key), '%s.%s' % (y, key))
                      for key, k in self.kw_kernels.items()])

        def gen_expr(self, x, y, theta_prefix=''):
            return Template('(${expr*})').render(
                expr=[k.gen_expr('%s.%s' % (x, key),
                                 '%s.%s' % (y, key),
                                 '%s%s.' % (theta_prefix, key))
                      for key, k in self.kw_kernels.items()])

        @property
        def theta(self):
            return tuple(k.theta for k in self.kw_kernels.values())

        @theta.setter
        def theta(self, seq):
            for kernel, value in zip(self.kw_kernels.values(), seq):
                kernel.theta = value

        @property
        def bounds(self):
            return tuple(k.bounds for k in self.kw_kernels.values())

    return TensorProductKernel(**kw_kernels)

# class Convolution(Kernel):
#     def __init__(self, kernel):
#         self.kernel = kernel

#     def __call__(self, object1, object2):
#         sum = 0.0
#         for part1 in object1:
#             for part2 in object2:
#                 sum += self.kernel(part1, part2)
#         return sum

#     def __repr__(self):
#         return 'ΣΣ{}'.format(repr(self.kernel))

#     @property
#     def theta(self):
#         return self.kernel.theta

#     def gen_constexpr(self, X, Y):
#         return ' + '.join([self.kernel.gen_constexpr(x, y)
#                            for x in X for y in Y])
