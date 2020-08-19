#!/usr/bin/env python
# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from collections import namedtuple, OrderedDict
import numpy as np
import sympy as sy
from sympy.utilities.lambdify import lambdify
from graphdot.codegen import Template
from graphdot.codegen.sympy_printer import cudacxxcode
from graphdot.codegen.cpptool import cpptype


class MicroKernel(ABC):
    '''The abstract base class for all microkernels.'''

    @property
    @abstractmethod
    def name(self):
        '''Name of the kernel.'''
        pass

    @abstractmethod
    def __call__(self, i, j, jac=False):
        '''Evaluates the kernel.

        Parameters
        ----------
        i, j: feature
            Inputs to the kernel.
        jac: Boolean
            Whether or not to return the gradient of the kernel with respect to
            kernel hyperparameters alongside the kernel value.

        Returns
        -------
        k_ij: scalar
            The value of the kernel as evaluated on i and j.
        jacobian: 1D ndarray
            The gradient of the kernel with regard to hyperparameters.
        '''
        pass

    @abstractmethod
    def __repr__(self):
        '''Evaluating the representation of a kernel should create an exact
        instance of the kernel itself.'''
        pass

    @abstractmethod
    def gen_expr(self, x, y, theta_scope=''):
        '''Generate the C++ expression for evaluating the kernel and its
        partial derivatives.

        Parameters
        ----------
        x, y: str
            Name of the input variables.
        theta_scope: str
            The scope in which the hyperparameters is located.

        Returns
        -------
        expr: str
            A C++ expression that evaluates the kernel.
        jac_expr: list of strs
            C++ expressions that evaluate the derivative of the kernel.
        '''
        pass

    @property
    @abstractmethod
    def theta(self):
        '''A tuple of all the kernel hyperparameters.'''
        pass

    @theta.setter
    @abstractmethod
    def theta(self, value):
        '''Method for setting the kernel hyperparameters from a tuple.'''
        pass

    @property
    @abstractmethod
    def bounds(self):
        '''A list of 2-tuples for the lower and upper bounds of each kernel
        hyperparameter.'''
        pass

    def _assert_bounds(self, hyp, bounds):
        if not ((isinstance(bounds, tuple) and len(bounds) == 2)
                or bounds == 'fixed'):
            raise ValueError(
                f'Bounds for hyperparameter {hyp} of kernel {self.name} must '
                f'be a 2-tuple or "fixed": {bounds} provided.'
            )

    @staticmethod
    def from_sympy(name, desc, expr, vars, *hyperparameter_specs):
        '''Create a pairwise kernel class from a SymPy expression.

        Parameters
        ----------
        name: str
            The name of the kernel. Must be a valid Python identifier.
        desc: str
            A human-readable description of the kernel. Will be used to build
            the docstring of the returned kernel class.
        expr: str or SymPy expression
            Expression of the kernel in SymPy format.
        vars: 2-tuple of str or SymPy symbols
            The input variables of the kernel as shown up in the expression.
            A kernel must have exactly 2 input variables. All other symbols
            that show up in its expression should be regarded as
            hyperparameters.
        hyperparameter_specs: list of hyperparameter specifications in one of
        the formats below:

            | symbol,
            | (symbol,),
            | (symbol, dtype),
            | (symbol, dtype, description),
            | (symbol, dtype, lower_bound, upper_bound),
            | (symbol, dtype, lower_bound, upper_bound, description),

            If a default set of lower and upper bounds are not defined here,
            then it must be specified explicitly during kernel object
            creation, using arguments as specified in the kernel class's
            docstring.
        '''
        return _from_sympy(name, desc, expr, vars, *hyperparameter_specs)

    def __add__(self, k):
        r"""Implements the additive kernel composition semantics, i.e.
        expression ``k1 + k2`` creates
        :math:`k_+(a, b) = k_1(a, b) + k_2(a, b)`"""
        return MicroKernelExpr.add(self, k)

    def __radd__(self, k):
        return MicroKernelExpr.add(k, self)

    def __mul__(self, k):
        r"""Implements the multiplicative kernel composition semantics, i.e.
        expression ``k1 * k2`` creates
        :math:`k_\times(a, b) = k_1(a, b) \times k_2(a, b)`"""
        return MicroKernelExpr.mul(self, k)

    def __rmul__(self, k):
        return MicroKernelExpr.mul(k, self)

    def __pow__(self, c):
        r"""Implements the exponentiation semantics, i.e.
        expression ``k1**c`` creates
        :math:`k_{exp}(a, b) = k_1(a, b) ** k_2(a, b)`"""
        return MicroKernelExpr.pow(self, c)


class MicroKernelExpr(MicroKernel):

    @property
    @abstractmethod
    def opstr(self):
        pass

    def __init__(self, k1, k2):
        self.k1 = k1
        self.k2 = k2

    def __repr__(self):
        return f'{repr(self.k1)} {self.opstr} {repr(self.k2)}'

    @property
    def theta(self):
        return (self.k1.theta, self.k2.theta)

    @theta.setter
    def theta(self, seq):
        self.k1.theta = seq[0]
        self.k2.theta = seq[1]

    @property
    def bounds(self):
        return (self.k1.bounds, self.k2.bounds)

    @staticmethod
    def add(k1, k2):
        k1 = Constant(k1) if np.isscalar(k1) else k1
        k2 = Constant(k2) if np.isscalar(k2) else k2

        @cpptype(k1=k1.dtype, k2=k2.dtype)
        class Add(MicroKernelExpr):

            @property
            def opstr(self):
                return '+'

            @property
            def name(self):
                return 'Add'

            def __call__(self, i, j, jac=False):
                if jac is True:
                    f1, J1 = self.k1(i, j, True)
                    f2, J2 = self.k2(i, j, True)
                    return (f1 + f2, J1 + J2)
                else:
                    return self.k1(i, j, False) + self.k2(i, j, False)

            def gen_expr(self, x, y, theta_scope=''):
                f1, J1 = self.k1.gen_expr(x, y, theta_scope + 'k1.')
                f2, J2 = self.k2.gen_expr(x, y, theta_scope + 'k2.')
                return (f'({f1} + {f2})', J1 + J2)

        return Add(k1, k2)

    @staticmethod
    def mul(k1, k2):
        k1 = Constant(k1) if np.isscalar(k1) else k1
        k2 = Constant(k2) if np.isscalar(k2) else k2

        @cpptype(k1=k1.dtype, k2=k2.dtype)
        class Multiply(MicroKernelExpr):

            @property
            def opstr(self):
                return '*'

            @property
            def name(self):
                return 'Multiply'

            def __call__(self, i, j, jac=False):
                if jac is True:
                    f1, J1 = self.k1(i, j, True)
                    f2, J2 = self.k2(i, j, True)
                    return (
                        f1 * f2,
                        np.array(
                            [j1 * f2 for j1 in J1] + [f1 * j2 for j2 in J2]
                        )
                    )
                else:
                    return self.k1(i, j, False) * self.k2(i, j, False)

            def gen_expr(self, x, y, theta_scope=''):
                f1, J1 = self.k1.gen_expr(x, y, theta_scope + 'k1.')
                f2, J2 = self.k2.gen_expr(x, y, theta_scope + 'k2.')
                return (
                    f'({f1} * {f2})',
                    [f'({j1} * {f2})' for j1 in J1] +
                    [f'({f1} * {j2})' for j2 in J2]
                )

        return Multiply(k1, k2)

    @staticmethod
    def pow(k1, c):
        if np.isscalar(c):
            k2 = Constant(c)
        elif isinstance(c, MicroKernel) and c.name == 'Constant':
            k2 = c
        else:
            raise ValueError(
                f'Exponent must be a constant or constant microkernel, '
                f'got {c} instead.'
            )

        @cpptype(k1=k1.dtype, k2=k2.dtype)
        class Exponentiation(MicroKernelExpr):

            @property
            def opstr(self):
                return '**'

            @property
            def name(self):
                return 'Exponentiation'

            def __call__(self, i, j, jac=False):
                if jac is True:
                    # d(x^y) / dx = y * x^(y - 1)
                    # d(x^y) / dy = x^y * log(x)
                    f1, J1 = self.k1(i, j, True)
                    f2, J2 = self.k2(i, j, True)
                    return (
                        f1**f2,
                        np.array(
                            [f2 * f1**(f2 - 1) * j1 for j1 in J1] +
                            [f1**f2 * np.log(f1) * j2 for j2 in J2]
                        )
                    )
                else:
                    return self.k1(i, j, False)**self.k2(i, j, False)

            def gen_expr(self, x, y, theta_scope=''):
                f1, J1 = self.k1.gen_expr(x, y, theta_scope + 'k1.')
                f2, J2 = self.k2.gen_expr(x, y, theta_scope + 'k2.')
                return (
                    f'__powf({f1}, {f2})',
                    [f'({f2} * __powf({f1}, {f2} - 1) * {j})' for j in J1] +
                    [f'(__powf({f1}, {f2}) * __logf({f1}) * {j})' for j in J2]
                )

        return Exponentiation(k1, k2)


def Constant(c, c_bounds=None):
    r"""Creates a no-op microkernel that returns a constant value,
    i.e. :math:`k_\mathrm{c}(\cdot, \cdot) \equiv constant`. This kernel is
    often mutliplied with other microkernels as an adjustable weight.

    Parameters
    ----------
    c: float > 0
        The constant value.
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

        def gen_expr(self, x, y, theta_scope=''):
            return f'{theta_scope}c', ['1.0f']

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


def _from_sympy(name, desc, expr, vars, *hyperparameter_specs):
    '''Create a microkernel class from a SymPy expression.

    Parameters
    ----------
    name: str
        The name of the microkernel. Must be a valid Python identifier.
    desc: str
        A human-readable description of the microkernel. Will be used to build
        the docstring of the returned microkernel class.
    expr: str or SymPy expression
        Expression of the microkernel in SymPy format.
    vars: 2-tuple of str or SymPy symbols
        The input variables of the microkernel as shown up in the expression.
        A microkernel must have exactly 2 input variables. All other symbols
        that show up in its expression should be regarded as
        hyperparameters.
    hyperparameter_specs: list of hyperparameter specifications in one of
    the formats below:

        | symbol,
        | (symbol,),
        | (symbol, dtype),
        | (symbol, dtype, description),
        | (symbol, dtype, lower_bound, upper_bound),
        | (symbol, dtype, lower_bound, upper_bound, description),

        If a default set of lower and upper bounds are not defined here,
        then it must be specified explicitly during microkernel object
        creation, using arguments as specified in the microkernel class's
        docstring.
    '''

    assert(isinstance(name, str) and name.isidentifier())

    '''parse expression'''
    if isinstance(expr, str):
        expr = sy.sympify(expr)

    '''check input variables'''
    if len(vars) != 2:
        raise ValueError('A microkernel must have exactly two variables')
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

    '''create microkernel class'''
    class CppType(type(MicroKernel)):
        @property
        def dtype(cls):
            return cls._dtype

    class uKernel(MicroKernel, metaclass=CppType):

        _expr = expr
        _vars = vars
        _hyperdefs = hyperdefs
        _dtype = np.dtype([(k, v['dtype']) for k, v in hyperdefs.items()],
                          align=True)

        @property
        def name(self):
            return name

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
                            f'Hyperparameter {symbol} not provided '
                            f'for {self.name}'
                        )

                try:
                    bounds[symbol] = kwargs['%s_bounds' % symbol]
                except KeyError:
                    try:
                        bounds[symbol] = self._hyperdefs[symbol]['bounds']
                    except KeyError:
                        raise KeyError(
                            f'Bounds for hyperparameter {symbol} of '
                            f'microkernel {self.name} not set, and '
                            f'no defaults were given.'
                        )
                self._assert_bounds(symbol, bounds[symbol])

        # @cached_property
        @property
        def _vars_and_hypers(self):
            if not hasattr(self, '_vars_and_hypers_cached'):
                self._vars_and_hypers_cached = [
                    *self._vars, *self._hyperdefs.keys()
                ]
            return self._vars_and_hypers_cached

        # @cached_property
        @property
        def _fun(self):
            if not hasattr(self, '_fun_cached'):
                self._fun_cached = lambdify(
                    self._vars_and_hypers,
                    self._expr
                )
            return self._fun_cached
            # return lambdify(self._vars_and_hypers, self._expr)

        # @cached_property
        @property
        def _jac(self):
            if not hasattr(self, '_jac_cached'):
                self._jac_cached = [
                    lambdify(self._vars_and_hypers, sy.diff(expr, h))
                    for h in self._hyperdefs
                ]
            return self._jac_cached
            # return [lambdify(self._vars_and_hypers, sy.diff(expr, h))
            #         for h in self._hyperdefs]

        def __call__(self, x1, x2, jac=False):
            if jac is True:
                return (
                    self._fun(x1, x2, *self.theta),
                    np.array([j(x1, x2, *self.theta) for j in self._jac])
                )
            else:
                return self._fun(x1, x2, *self.theta)

        def __repr__(self):
            return Template('${cls}(${theta, }, ${bounds, })').render(
                cls=self.name,
                theta=[f'{n}={v}' for n, v in self._theta_values.items()],
                bounds=[f'{n}_bounds={v}'
                        for n, v in self._theta_bounds.items()]
            )

        def gen_expr(self, x, y, theta_scope=''):
            nmap = {
                str(self._vars[0]): x,
                str(self._vars[1]): y,
                **{t: theta_scope + t for t in self._hyperdefs}
            }

            return (
                cudacxxcode(self._expr, nmap),
                [cudacxxcode(sy.diff(self._expr, h), nmap)
                 for h in self._hyperdefs]
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
                f'{self.name}Hyperparameters',
                self._theta_values.keys()
            )(**self._theta_values)

        @theta.setter
        def theta(self, seq):
            assert(len(seq) == len(self._theta_values))
            for theta, value in zip(self._hyperdefs, seq):
                self._theta_values[theta] = value

        @property
        def bounds(self):
            return tuple(self._theta_bounds.values())

    '''furnish doc strings'''
    param_docs = [
        Template(
            '${name}: ${type}\n'
            '    ${desc\n    }\n'
            '${name}_bounds: tuple or "fixed"\n'
            '    Lower and upper bounds of `${name}` with respect to '
            'hyperparameter optimization. If "fixed", the hyperparameter will '
            'not be optimized during training.'
        ).render(
            name=name,
            type=hdef['dtype'],
            desc=[s.strip() for s in hdef.get('doc', '').split('\n')]
        ) for name, hdef in hyperdefs.items()
    ]

    uKernel.__doc__ = Template(
        '${desc}\n'
        '\n'
        'Parameters\n'
        '----------\n'
        '${param_docs\n}',
        escape=False
    ).render(
        desc='\n'.join([s.strip() for s in desc.split('\n')]),
        param_docs=param_docs
    )

    return uKernel
