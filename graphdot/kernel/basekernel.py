#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module defines base kernels and composibility rules for creating vertex
and edge kernels for the marginalized graph kernel.
"""
from collections import namedtuple, OrderedDict
import numpy as np
import sympy as sy
from sympy.utilities.autowrap import ufuncify
from graphdot.codegen import Template
from graphdot.codegen.typetool import cpptype
from graphdot.codegen.sympy_printer import cudacxxcode

__all__ = ['BaseKernel',
           'Constant',
           'KroneckerDelta',
           'SquareExponential',
           'RationalQuadratic',
           'TensorProduct']


class BaseKernel:
    """
    Parent class for all base kernels
    """

    @staticmethod
    def create(kernel, desc, expr, vars, *hyperparameter_specs):

        '''parse expression'''
        if isinstance(expr, str):
            expr = sy.sympify(expr)

        '''check input variables'''
        if len(vars) != 2:
            raise ValueError('A kernel must have exactly two variables')
        vars = [sy.Symbol(v) if isinstance(v, str) else v for v in vars]

        '''parse the list of hyperparameters'''
        hyperdefs = OrderedDict()
        for spec in hyperparameter_specs:
            if not hasattr(spec, '__iter__'):
                symbol = spec
                hyperdefs[symbol] = dict(dtype=np.dtype(np.float32))
            if len(spec) == 1:
                symbol = spec[0]
                hyperdefs[symbol] = dict(dtype=np.dtype(np.float32))
            if len(spec) == 2:
                symbol, dtype = spec
                hyperdefs[symbol] = dict(dtype=np.dtype(dtype))
            if len(spec) == 3:
                symbol, dtype, doc = spec
                hyperdefs[symbol] = dict(dtype=np.dtype(dtype), doc=doc)
            elif len(spec) == 4:
                symbol, dtype, lb, ub = spec
                hyperdefs[symbol] = dict(dtype=np.dtype(dtype),
                                         bounds=(lb, ub))
            elif len(spec) == 5:
                symbol, dtype, lb, ub, doc = spec
                hyperdefs[symbol] = dict(dtype=np.dtype(dtype),
                                         bounds=(lb, ub),
                                         doc=doc)
            else:
                raise ValueError(
                    'Invalid hyperparameter specification, '
                    'must be one of\n'
                    '(symbol)\n',
                    '(symbol, dtype)\n',
                    '(symbol, dtype, doc)\n',
                    '(symbol, dtype, lb, ub)\n',
                    '(symbol, dtype, lb, ub, doc)\n',
                )

        '''create kernel class'''
        class CppType(type):
            @property
            def dtype(cls):
                return cls._dtype

        class Kernel(BaseKernel, metaclass=CppType):

            __name__ = kernel

            _expr = expr
            _vars = vars
            _hyperdefs = hyperdefs
            _dtype = np.dtype([(k, v['dtype']) for k, v in hyperdefs.items()],
                              align=True)

            def __init__(self, *args, **kwargs):

                self._theta_values = values = OrderedDict()
                self._theta_bounds = bounds = OrderedDict()

                for symbol, value in zip(self._hyperdefs, args):
                    values[symbol] = value

                for symbol in self._hyperdefs:
                    try:
                        values[symbol] = kwargs[symbol]
                    except KeyError:
                        if symbol not in values:
                            raise KeyError(
                                'Hyperparameter {} not provided for {}'.format(
                                    symbol,
                                    self.__name__
                                )
                            )

                    try:
                        bounds[symbol] = kwargs['%s_bounds' % symbol]
                    except KeyError:
                        try:
                            bounds[symbol] = self._hyperdefs[symbol]['bounds']
                        except KeyError:
                            raise KeyError(
                                'Bounds for hyperparameter {} of kernel {} not'
                                ' set, and no default value exists.'.format(
                                    symbol,
                                    self.__name__
                                )
                            )

            def __call__(self, x1, x2):
                return self.ufunc.outer(x1, x2)

            def __repr__(self):
                return Template('${cls}(${theta, }, ${bounds, })').render(
                    cls=self.__name__,
                    theta=['{}={}'.format(*v)
                           for v in self._theta_values.items()],
                    bounds=['{}_bounds={}'.format(*v)
                            for v in self._theta_bounds.items()]
                )

            @property
            def _bound_expr(self):
                return self._expr.subs(self._theta_values.items())

            @property
            def ufunc(self):
                if not hasattr(self, '_ufunc'):
                    self._ufunc = ufuncify(self._vars, self._bound_expr)
                return self._ufunc

            def gen_constexpr(self, x, y):
                return cudacxxcode(
                    self._bound_expr,
                    {str(self._vars[0]): x,
                     str(self._vars[1]): y}
                )

            def gen_expr(self, x, y, theta_scope=''):
                return cudacxxcode(
                    self._expr,
                    {str(self._vars[0]): x,
                     str(self._vars[1]): y,
                     **{t: theta_scope + t for t in self._hyperdefs}}
                )

            @property
            def dtype(self):
                return self._dtype

            @property
            def state(self):
                return tuple(self._theta_values.values())

            @property
            def theta(self):
                return namedtuple(
                    self.__name__ + 'Hyperparameters',
                    self._theta_values.keys()
                )(**self._theta_values)

            @theta.setter
            def theta(self, seq):
                assert(len(seq) == len(self._theta_values))
                for theta, value in zip(self._hyperdefs, seq):
                    self._theta_values[theta] = value
                if hasattr(self, '_ufunc'):
                    del self._ufunc

            @property
            def bounds(self):
                return tuple(self._theta_bounds.values())

        '''furnish doc strings'''
        param_docs = [
            Template(
                '${name}: ${type}\n'
                '    ${desc\n    }\n'
                '${name}_bounds: pair of ${type}\n'
                '    Lower and upper bounds of `${name}`.'
            ).render(
                name=name,
                type=str(hdef['dtype']),
                desc=[s.strip() for s in hdef.get('doc', '').split('\n')]
            ) for name, hdef in hyperdefs.items()
        ]

        Kernel.__doc__ = Template(
            '${desc}\n'
            '\n'
            'Parameters\n'
            '----------\n'
            '${param_docs\n}',
            escape_repl=False
        ).render(
            desc='\n'.join([s.strip() for s in desc.split('\n')]),
            param_docs=param_docs
        )

        return Kernel

    def __add__(self, k):
        r"""Implements the additive kernel composition semantics, i.e.
        expression ``k1 + k2`` creates
        :math:`k_+(a, b) = k_1(a, b) + k_2(a, b)`"""
        return BaseKernel._op(
            self,
            k if isinstance(k, BaseKernel) else Constant(k),
            lambda x, y: x + y, '+'
        )

    def __radd__(self, k):
        return BaseKernel._op(
            k if isinstance(k, BaseKernel) else Constant(k),
            self,
            lambda x, y: x + y, '+'
        )

    def __mul__(self, k):
        r"""Implements the multiplicative kernel composition semantics, i.e.
        expression ``k1 * k2`` creates
        :math:`k_\times(a, b) = k_1(a, b) \times k_2(a, b)`"""
        return BaseKernel._op(
            self,
            k if isinstance(k, BaseKernel) else Constant(k),
            lambda x, y: x * y, '*'
        )

    def __rmul__(self, k):
        return BaseKernel._op(
            k if isinstance(k, BaseKernel) else Constant(k),
            self,
            lambda x, y: x * y, '*'
        )

    @staticmethod
    def _op(k1, k2, op, opstr):
        # only works with python >= 3.6
        # @cpptype(k1=k1.dtype, k2=k2.dtype)
        @cpptype([('k1', k1.dtype), ('k2', k2.dtype)])
        class KernelOperator(BaseKernel):
            def __init__(self, k1, k2):
                self.k1 = k1
                self.k2 = k2

            def __call__(self, i, j):
                return op(self.k1(i, j), self.k2(i, j))

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


@cpptype([])
class _Multiply(BaseKernel):
    """
    Direct product is technically not a valid base kernel
    Only used internally for treating weighted edges
    """

    def __call__(self, x1, x2):
        return x1 * x2

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


def Constant(c, c_bounds=(0, np.inf)):
    r"""Creates a no-op kernel that returns a constant value (often being 1),
    i.e. :math:`k_\mathrm{c}(\cdot, \cdot) \equiv constant`

    Parameters
    ----------
    constant: float > 0
        The value of the kernel

    Returns
    -------
    BaseKernel
        A kernel instance of corresponding behavior
    """

    # only works with python >= 3.6
    # @cpptype(constant=np.float32)
    @cpptype([('c', np.float32)])
    class ConstantKernel(BaseKernel):
        def __init__(self, c, c_bounds):
            self.c = float(c)
            self.c_bounds = c_bounds

        def __call__(self, i, j):
            return self.c

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
    BaseKernel
        A kernel instance of corresponding behavior
    """

    # only works with python >= 3.6
    # @cpptype(lo=np.float32, hi=np.float32)
    @cpptype([('h', np.float32)])
    class KroneckerDeltaKernel(BaseKernel):

        def __init__(self, h, h_bounds):
            self.h = float(h)
            self.h_bounds = h_bounds

        def __call__(self, i, j):
            return 1.0 if i == j else self.h

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


SquareExponential = BaseKernel.create(
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

RationalQuadratic = BaseKernel.create(
    'RationalQuadratic',

    r"""A rational quadratic kernel is equivalent to the sum of many square
    exponential kernels with different length scales. The parameter `alpha`
    tunes the relative weights between large and small length scales. When
    alpha approaches infinity, the kernel is identical to the square
    exponential kernel.""",

    '(1 + (x - y)**2/(2 * alpha * length_scale**2))**(-alpha)',

    ('x', 'y'),

    ('length_scale', np.float32, 1e-6, np.inf,
     r"""The smallest length scale of the square exponential components."""),
    ('alpha', np.float32, 1e-3, np.inf,
     r"""The relative weights of large-scale square exponential components.
     Larger alpha values leads to a faster decay of the weights for larger
     length scales.""")
)
r"""A rational quadratic kernel
"""


# def Product(*kernels):
#     @cpptype([('k%d' % i, ker.dtype) for i, ker in enumerate(kernels)])
#     class ProductKernel(BaseKernel):
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
    class TensorProductKernel(BaseKernel):
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

# class Convolution(BaseKernel):
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
