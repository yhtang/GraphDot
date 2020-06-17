#!/usr/bin/env python
# -*- coding: utf-8 -*-
from collections import OrderedDict
import numpy as np
import scipy.spatial.distance
import sympy
from sympy.core.sympify import sympify
from sympy.utilities.autowrap import ufuncify


class RBFKernel:

    def __init__(self, expr, x, **hyperparameters):
        self.expr = sympify(expr)
        self._params = OrderedDict(**hyperparameters)
        self._param = (
            sympy.symbols(x),
            *sympy.symbols(','.join(self._params.keys()))
        )
        self._fun = ufuncify(self._param, self.expr)
        self._grad = [ufuncify(self._param, sympy.diff(self.expr, t))
                      for t in self._params]

    def get_params(self):
        return self._params

    @property
    def theta(self):
        return np.log(list(self._params.values()))

    @theta.setter
    def theta(self, args):
        for k, v in zip(self._params, np.exp(args)):
            self._params[k] = v

    def __call__(self, X, Y=None):
        if Y is None:
            d = scipy.spatial.distance.cdist(X, X)
        else:
            d = scipy.spatial.distance.cdist(X, Y)
        return self._fun(d, *self._params.values())

    def gradient(self, X):
        d = scipy.spatial.distance.cdist(X, X)
        return [g(d, *self._params.values()) for g in self._grad]

    def diag(self, X):
        return self._fun(np.zeros(len(X)), *self._params.values())
