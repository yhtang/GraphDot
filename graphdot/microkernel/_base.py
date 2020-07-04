#!/usr/bin/env python
# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from ._from_sympy import _from_sympy
from ._expr import MicroKernelExpr


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
    def gen_expr(self, x, y, jac=False, theta_scope=''):
        '''Generate the C++ expression for evaluating the kernel.

        Parameters
        ----------
        x, y: str
            Name of the input variables.
        jac: Boolean
            Whether or not to return a list of expressions that evaluate the
            gradient of the kernel with respect to hyperparameters alongside
            the kernel value.
        theta_scope: str
            The scope in which the hyperparameters is located.

        Returns
        -------
        expr: str
            A C++ expression that evaluates the kernel.
        jac_expr: list of strs (optional, only if jac is True)
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
        if not isinstance(bounds, tuple) or len(bounds) != 2:
            raise ValueError(
                f'Bounds for hyperparameter {hyp} of kernel {self.name} must '
                f'be a 2-tuple: {bounds} provided.'
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
