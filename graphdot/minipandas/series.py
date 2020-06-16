#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from graphdot.codegen.typetool import common_concrete_type, common_min_type


class Series(np.ndarray):
    '''A thin wrapper to customize serialization behavior'''
    def __new__(cls, input):
        if isinstance(input, np.ndarray):
            series = input.view(cls)
            if np.issctype(series.dtype):
                series._concrete_type = series.dtype
            else:
                series._concrete_type = common_concrete_type.of_values(input)
        else:
            t = common_min_type.of_values(input)
            dtype = np.dtype(t) if np.issctype(t) else np.dtype(np.object)
            series = np.empty(len(input), dtype=dtype).view(cls)  # ensures 1D
            series[:] = input
            series._concrete_type = t
        return series

    def __repr__(self):
        return np.array2string(self, separator=',', max_line_width=1e20)

    @property
    def concrete_type(self):
        return self._concrete_type

    def __reduce__(self):
        recon, args, state = super(Series, self).__reduce__()
        return (recon, args, (state, self.__dict__))

    def __setstate__(self, states):
        state, dict_ = states
        self.__dict__.update(**dict_)
        super(Series, self).__setstate__(state)
