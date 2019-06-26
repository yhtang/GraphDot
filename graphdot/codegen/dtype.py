#!/usr/bin/env python
# -*- coding: utf-8 -*-
from graphdot.codegen import Template
import numpy as np


def decltype(type):
    type = np.dtype(type, align=True)  # convert np.float32 etc. to dtype
    if type.names is not None:
        return Template(r'''struct{${members;};}''').render(
            members=['{} {}'.format(decltype(t), v)
                     for v, (t, offset) in type.fields.items()])
    # elif type.subdtype is not None:
    #     return Template(r'''${type} ${name}${dim}''').render(
    #         type=type.name, name=
    #     )
    else:
        return str(type.name)


def rowtype(df):
    order = np.argsort([df.dtypes[key].itemsize for key in df.columns])
    packed_attributes = [df.columns[i] for i in order[-1::-1]]
    packed_dtype = np.dtype([(key, df.dtypes[key].newbyteorder('='))
                             for key in packed_attributes], align=True)
    return packed_dtype
