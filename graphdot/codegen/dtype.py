#!/usr/bin/env python
# -*- coding: utf-8 -*-
from graphdot.codegen import Template
import numpy


def decltype(type):
    type = numpy.dtype(type, align=True)  # convert numpy.float32 etc. to dtype
    if type.names is not None:
        return Template('''struct{${members;};}''').render(
            members=['{} {}'.format(decltype(t), v)
                     for v, (t, offset) in type.fields.items()])
    else:
        return str(type.name)
