from graphdot.marginalized.basekernel import Constant
from graphdot.marginalized.basekernel import KroneckerDelta
from graphdot.marginalized.basekernel import SquareExponential

import random
import pytest

kernels = [
    Constant(1.0),
    KroneckerDelta(0.5),
    SquareExponential(1.0)
]
