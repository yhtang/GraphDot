#!/usr/bin/env python
# -*- coding: utf-8 -*-
from sympy.codegen import ast
from sympy.printing.cxxcode import CXX11CodePrinter


class CUDACXX11CodePrinter(CXX11CodePrinter):
    _ns = ''

    def __call__(self, expr):
        return self.doprint(expr)


cuda_cxx11_code_printer = CUDACXX11CodePrinter(
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
