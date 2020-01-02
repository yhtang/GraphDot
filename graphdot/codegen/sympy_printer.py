#!/usr/bin/env python
# -*- coding: utf-8 -*-
from sympy.codegen import ast
from sympy.printing.cxxcode import CXX11CodePrinter


class CUDACXX11CodePrinter(CXX11CodePrinter):
    _ns = ''

    def __init__(self, settings):
        super().__init__(settings)

    def __call__(self, expr, symbol_to_variable):
        self.symbol_to_variable = symbol_to_variable
        return self.doprint(expr)

    def _print_Symbol(self, expr):
        name = self.symbol_to_variable[super()._print_Symbol(expr)]
        if expr in self._settings['dereference']:
            return '(*{0})'.format(name)
        else:
            return name


cudacxxcode = CUDACXX11CodePrinter(
    dict(
        user_functions={
            'Pow': [
                # if exp is positive integer
                (lambda b, e: e.is_integer and int(e) >= 0,
                 lambda b, e: 'graphdot::ipow<%d>(%s)' % (int(e), b)),
                # if exp is negative integer
                (lambda b, e: e.is_integer and int(e) < 0,
                 lambda b, e: 'graphdot::ripow<%d>(%s)' % (-int(e), b)),
                # otherwise
                (lambda b, e: True, 'powf'),
            ]
        },
        type_aliases={
            ast.real: ast.float32,
            ast.integer: ast.int32
        }
    )
)
