#!/usr/bin/env python
# -*- coding: utf-8 -*-
from collections import namedtuple, OrderedDict
import numpy as np
import sympy as sy
from sympy.utilities.lambdify import lambdify
from graphdot.codegen import Template
from graphdot.codegen.sympy_printer import cudacxxcode
from ._base import MicroKernel


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
    class CppType(type):
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

        def gen_expr(self, x, y, jac=False, theta_scope=''):
            nmap = {
                str(self._vars[0]): x,
                str(self._vars[1]): y,
                **{t: theta_scope + t for t in self._hyperdefs}
            }

            if jac is True:
                return (
                    cudacxxcode(self._expr, nmap),
                    [cudacxxcode(sy.diff(self._expr, h), nmap)
                        for h in self._hyperdefs]
                )
            else:
                return cudacxxcode(self._expr, nmap)

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
            '${name}_bounds: pair of ${type}\n'
            '    Lower and upper bounds of `${name}`.'
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
