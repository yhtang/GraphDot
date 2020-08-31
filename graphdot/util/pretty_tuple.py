#!/usr/bin/env python
# -*- coding: utf-8 -*-
from collections import namedtuple
from graphdot.codegen.template import Template


def pretty_tuple(typename, fields):

    class PrettyTuple(namedtuple(typename, fields)):

        def __repr__(self):
            out = []
            for f in fields:
                if hasattr(getattr(self, f), '__iter__'):
                    out.append(
                        Template(
                            '${field} : ${typename}\n\t${lines\n\t}'
                        ).render(
                            field=f,
                            typename=type(getattr(self, f)).__name__,
                            lines=repr(getattr(self, f)).split('\n')
                        )
                    )
                else:
                    out.append(f'{f} : {repr(getattr(self, f))}')
            return '\n'.join(out)

    PrettyTuple.__name__ = typename

    return PrettyTuple
