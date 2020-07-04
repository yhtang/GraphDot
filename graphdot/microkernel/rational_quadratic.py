#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from ._base import MicroKernel


RationalQuadratic = MicroKernel.from_sympy(
    'RationalQuadratic',

    r"""A rational quadratic kernel is equivalent to the sum of many square
    exponential kernels with different length scales. The parameter `alpha`
    tunes the relative weights between large and small length scales. When
    alpha approaches infinity, the kernel is identical to the square
    exponential kernel.""",

    '(1 + (x - y)**2 / (2 * alpha * length_scale**2))**(-alpha)',

    ('x', 'y'),

    ('length_scale', np.float32, 1e-6, np.inf,
     r"""The smallest length scale of the square exponential components."""),
    ('alpha', np.float32, 1e-3, np.inf,
     r"""The relative weights of large-scale square exponential components.
     Larger alpha values leads to a faster decay of the weights for larger
     length scales.""")
)
