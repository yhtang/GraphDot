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
            'exp': '__expf',
            'pow': '__powf',
        },
        type_aliases={
            ast.real: ast.float32,
            ast.integer: ast.int32
        }
    )
)
