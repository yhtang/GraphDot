#!/usr/bin/env python
# -*- coding: utf-8 -*-
from collections import OrderedDict
import numpy as np
import sympy
from sympy.core.sympify import sympify
from sympy.utilities.autowrap import ufuncify
from graphdot.util.pretty_tuple import pretty_tuple


class KernelOverMetric:

    def __init__(self, distance, expr, x, **hyperparameters):
        self._init_args = (expr, x)
        self._init_kwargs = hyperparameters
        self.distance = distance
        self.expr = sympify(expr)
        self._hyperparams = OrderedDict()
        self._hyperbounds = OrderedDict()
        for key, val in hyperparameters.items():
            if not hasattr(val, '__iter__'):
                self._hyperparams[key] = val
                self._hyperbounds[key] = (0, np.inf)
            elif len(val) == 1:
                self._hyperparams[key] = val[0]
                self._hyperbounds[key] = (0, np.inf)
            elif len(val) == 2:
                self._hyperparams[key] = val[0]
                self._hyperbounds[key] = val[1]
            elif len(val) == 3:
                self._hyperparams[key] = val[0]
                self._hyperbounds[key] = (val[1], val[2])
        self.x = x
        vars = (
            sympy.symbols(x),
            *[sympy.symbols(h) for h in self._hyperparams]
        )
        self._fun = ufuncify(vars, self.expr)
        self._grad = [ufuncify(vars, sympy.diff(self.expr, t))
                      for t in self._hyperparams]
        self._grad_m = ufuncify(vars, sympy.diff(self.expr, sympy.symbols(x)))

    def __call__(self, X, Y=None, eval_gradient=False):
        if eval_gradient is False:
            return self._gramian(self.distance(X, Y))
        else:
            M, dM = self.distance(X, Y, eval_gradient=True)
            # must happen before _gramian since the latter is destructive
            grad = np.empty((*M.shape, len(self.theta)), order='F')
            for i, g in enumerate(self._grad):
                g(M, *self._hyperparams.values(), out=grad[:, :, i])
            if len(self.distance.theta) > 0:
                self._grad_m(
                    M, *self._hyperparams.values(),
                    out=grad[:, :, -1]
                )
                grad[:, :, len(self._grad):-1] = grad[:, :, [-1]]
                np.multiply(
                    grad[:, :, len(self._grad):],
                    dM,
                    out=grad[:, :, len(self._grad):]
                )
            return self._gramian(M), grad

    def _gramian(self, d):
        return self._fun(d, *self._hyperparams.values(), out=d)

    def diag(self, X):
        return self._fun(np.zeros(len(X)), *self._hyperparams.values())

    def get_params(self):
        return self._hyperparams

    @property
    def theta(self):
        return np.concatenate((
            np.log(list(self._hyperparams.values())),
            self.distance.theta
        ))

    @theta.setter
    def theta(self, args):
        for k, v in zip(self._hyperparams, np.exp(args)):
            self._hyperparams[k] = v
        self.distance.theta = args[len(self._hyperparams):]

    @property
    def bounds(self):
        return np.vstack((
            np.log(np.vstack(self._hyperbounds.values())),
            self.distance.bounds
        ))

    @property
    def hyperparameters(self):
        return pretty_tuple(
            'RBFKernel',
            list(self._hyperparams.keys()) + ['distance']
        )(
            *self._hyperparams.values(),
            self.distance.hyperparameters
        )

    def clone_with_theta(self, theta=None):
        if theta is None:
            theta = self.theta
        k = type(self)(self.distance.clone_with_theta(),
                       *self._init_args, **self._init_kwargs)
        k.theta = theta
        return k
