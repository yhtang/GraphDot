#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from ._base import MicroKernel


SquareExponential = MicroKernel.from_sympy(
    'SquareExponential',

    r"""A square exponential kernel smoothly transitions from 1 to
    0 as the distance between two vectors increases from zero to infinity, i.e.
    :math:`k_\mathrm{se}(\mathbf{x}, \mathbf{y}) = \exp(-\frac{1}{2}
    \frac{\lVert \mathbf{x} - \mathbf{y} \rVert^2}{\sigma^2})`""",

    'exp(-0.5 * (x - y)**2 * length_scale**-2)',

    ('x', 'y'),

    ('length_scale', np.float32, 1e-6, np.inf,
     r"""Determines how quickly should the kernel decay to zero. The kernel has
     a value of approx. 0.606 at one length scale, 0.135 at two length
     scales, and 0.011 at three length scales.""")
)
