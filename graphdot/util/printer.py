#!/usr/bin/env python
# -*- coding: utf-8 -*-
def center(s, width):
    nspace = width - len(s)
    lspace = nspace // 2
    rspace = nspace - lspace
    return ' ' * lspace + s + ' ' * rspace


class markdown:

    _print_table_header = False

    @classmethod
    def table_start(cls):
        cls._print_table_header = True

    @classmethod
    def table_header(cls, *fields):
        strs = [fmt % value for _, fmt, value in fields]
        align = ['-' if fmt.startswith('%-') else '' for _, fmt, _ in fields]
        fmts = [f'%{a}{len(s)}s' for a, s in zip(align, strs)]
        header = '|'.join([fmt % f[0] for f, fmt in zip(fields, fmts)])
        sep = '|'.join(['-' * len(s) for s in strs])
        print(f'|{header}|\n|{sep}|')

    @classmethod
    def table(cls, *fields):
        if cls._print_table_header is True:
            cls.table_header(*fields)
            cls._print_table_header = False
        line = '|'.join([fmt % value for _, fmt, value in fields])
        print(f'|{line}|')
