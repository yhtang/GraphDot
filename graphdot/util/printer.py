#!/usr/bin/env python
# -*- coding: utf-8 -*-


class markdown:

    _print_table_header = False

    @classmethod
    def table_start(cls):
        cls._print_table_header = True

    @classmethod
    def table_header(cls, *fields):
        '''Print the header row of a Markdown table.

        Parameters
        ----------
        fields: list of (title, format, value) tuples
            Each tuple sets the title, format, and a dummy value for each
            column.
        '''
        # format content
        strs = [fmt % value for _, fmt, value in fields]
        # copy alignment specifier to column titles
        align = ['-' if fmt.startswith('%-') else '' for _, fmt, _ in fields]
        # format column titles
        fmts = [f'%{a}{len(s)}s' for a, s in zip(align, strs)]
        header = '|'.join([fmt % f[0] for f, fmt in zip(fields, fmts)])
        # join titles to form a line
        sep = '|'.join(['-' * len(s) for s in strs])
        print(f'|{header}|\n|{sep}|')

    @classmethod
    def table(cls, *fields, print_header='auto'):
        '''Print a row of data in Markdown table format.

        Parameters
        ----------
        fields: list of (title, format, value) tuples
            Each tuple sets the title of the associated column, format, and
            the value.
        print_header: Boolean or 'auto'
            If 'auto', a header row will automatically be printed if it is the
            first invocation of this method since calling
            :py:`markdown.table_start()`. If print_header is a boolean, a
            header row will be printed according to its truth value.
        '''
        if print_header is True or (print_header == 'auto' and
                                    cls._print_table_header is True):
            cls.table_header(*fields)
            cls._print_table_header = False
        line = '|'.join([fmt % value for _, fmt, value in fields])
        print(f'|{line}|')
