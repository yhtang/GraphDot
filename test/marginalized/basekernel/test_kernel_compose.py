from graphdot.marginalized.basekernel import Constant
from graphdot.marginalized.basekernel import KroneckerDelta
from graphdot.marginalized.basekernel import SquareExponential
from graphdot.marginalized.basekernel import Convolution
from graphdot.marginalized.basekernel import TensorProduct

import random
import pytest

inf = float('inf')
kernels = [
    Constant(1.0),
    KroneckerDelta(0.5),
    SquareExponential(1.0)
]
