#!/usr/bin/env python
# -*- coding: utf-8 -*-
from collections import namedtuple
import numpy as np
import copy


class Normalization:
    r"""Normalizes a kernel using the cosine of angle formula:
    :math:`k_\mathrm{normalized}(x, y) =
    \frac{k(x, y)}{\sqrt{k(x, x)k(y, y)}}`.

    Parameters
    ----------
    kernel: object
        The kernel to be normalized.
    """
    def __init__(self, kernel):
        self.kernel = kernel

    def __call__(self, X, Y=None, eval_gradient=False, **options):
        """Normalized outcome of
        :py:`self.kernel(X, Y, eval_gradient, **options)`.

        Parameters
        ----------
        Inherits that of the graph kernel object.

        Returns
        -------
        Inherits that of the graph kernel object.
        """
        if eval_gradient is True:
            R, dR = self.kernel(X, Y, eval_gradient=True, **options)
            if Y is None:
                ldiag = rdiag = R.diagonal()
            else:
                ldiag, ldDiag = self.kernel.diag(X, True, **options)
                rdiag, rdDiag = self.kernel.diag(Y, True, **options)
            ldiag_inv = 1 / ldiag
            rdiag_inv = 1 / rdiag
            ldiag_rsqrt = np.sqrt(ldiag_inv)
            rdiag_rsqrt = np.sqrt(rdiag_inv)
            K = ldiag_rsqrt[:, None] * R * rdiag_rsqrt[None, :]
            dK = []
            for i in range(dR.shape[-1]):
                dr = dR[:, :, i]
                if Y is None:
                    lddiag = rddiag = dr.diagonal()
                else:
                    lddiag = ldDiag[:, i]
                    rddiag = rdDiag[:, i]
                dk = (
                    ldiag_rsqrt[:, None] * dr * rdiag_rsqrt[None, :]
                    - 0.5 * ldiag_inv[:, None] * K * rdiag_inv[None, :] * (
                        np.outer(lddiag, rdiag) + np.outer(ldiag, rddiag)
                    )
                )
                dK.append(dk)
            dK = np.asfortranarray(np.stack(dK, axis=2))
            return K, dK
        else:
            R = self.kernel(X, Y, **options)
            if Y is None:
                ldiag = rdiag = R.diagonal()
            else:
                ldiag = self.kernel.diag(X, **options)
                rdiag = self.kernel.diag(Y, **options)
            ldiag_inv = 1 / ldiag
            rdiag_inv = 1 / rdiag
            ldiag_rsqrt = np.sqrt(ldiag_inv)
            rdiag_rsqrt = np.sqrt(rdiag_inv)
            K = ldiag_rsqrt[:, None] * R * rdiag_rsqrt[None, :]
            return K

    def diag(self, X, eval_gradient=False, **options):
        """Normalized outcome of :py:`self.kernel.diag(X, **options)`.

        Parameters
        ----------
        Inherits that of the graph kernel object.

        Returns
        -------
        Inherits that of the graph kernel object.
        """
        if eval_gradient is True:
            return np.ones(len(X)), np.ones((len(X), len(self.kernel.theta)))
        else:
            return np.ones(len(X))

    """⭣⭣⭣⭣⭣ scikit-learn interoperability methods ⭣⭣⭣⭣⭣"""

    @property
    def hyperparameters(self):
        return self.kernel.hyperparameters

    @property
    def theta(self):
        return self.kernel.theta

    @theta.setter
    def theta(self, value):
        self.kernel.theta = value

    @property
    def hyperparameter_bounds(self):
        return self.kernel.hyperparameter_bounds

    @property
    def bounds(self):
        return self.kernel.bounds

    def clone_with_theta(self, theta):
        clone = copy.deepcopy(self)
        clone.theta = theta
        return clone


class Exponentiation:
    r"""Raises a kernel to some exponentiation.
    :math:`k_\mathrm{exponentiation}(x, y) = k(x, y)^\xi`.

    Parameters
    ----------
    kernel: object
        The graph kernel to be exponentiated.
    xi: float
        The exponent to be raises.
    xi_bounds: (float, float)
        The range of the exponents to be searched during hyperparameter
        optimization.
    """
    def __init__(self, kernel, xi=1.0, xi_bounds=(0.1, 20.0)):
        self.kernel = kernel
        self.xi = xi
        self.xi_bounds = xi_bounds

    def __call__(self, X, Y=None, eval_gradient=False, **options):
        """Normalized outcome of
        :py:`self.kernel(X, Y, eval_gradient, **options)`.

        Parameters
        ----------
        Inherits that of the graph kernel object.

        Returns
        -------
        Inherits that of the graph kernel object.
        """
        if eval_gradient is True:
            R, dR = self.kernel(X, Y, eval_gradient=True, **options)
            K = R**self.xi
            dK = []
            dK.append(K * np.log(R))  # \frac{d R^\xi}{d \xi} = R^\xi \log R
            KK = self.xi * R**(self.xi - 1)
            # \frac{d R^\xi}{d \theta} = \xi R^(\xi - 1) \frac{d R}{d \theta}
            for i in range(dR.shape[-1]):
                dK.append(KK * dR[:, :, i])
            dK = np.stack(dK, axis=2)
            return K, dK
        else:
            R = self.kernel(X, Y, **options)
            return R**self.xi

    def diag(self, X, **options):
        """Normalized outcome of :py:`self.kernel.diag(X, **options)`.

        Parameters
        ----------
        Inherits that of the graph kernel object.

        Returns
        -------
        Inherits that of the graph kernel object.
        """
        return self.kernel.diag(X, **options)**self.xi

    """⭣⭣⭣⭣⭣ scikit-learn interoperability methods ⭣⭣⭣⭣⭣"""

    @property
    def hyperparameters(self):
        return namedtuple('ExponentiationHyperparameters', ['xi', 'kernel'])(
            self.xi, self.kernel.hyperparameters
        )

    @property
    def theta(self):
        return np.concatenate((np.log([self.xi]), self.kernel.theta))

    @theta.setter
    def theta(self, value):
        self.xi = np.exp(value[0])
        self.kernel.theta = value[1:]

    @property
    def hyperparameter_bounds(self):
        return namedtuple(
            'ExponentiationHyperparameterBounds',
            ['xi', 'kernel']
        )(
            self.xi_bounds, self.kernel.hyperparameter_bounds
        )

    @property
    def bounds(self):
        return np.vstack((
            np.log([self.xi_bounds]),
            self.kernel.bounds
        ))

    def clone_with_theta(self, theta):
        clone = copy.deepcopy(self)
        clone.theta = theta
        return clone
