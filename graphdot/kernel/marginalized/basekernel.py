#!/usr/bin/env python
# -*- coding: utf-8 -*-
import warnings
from graphdot.microkernel import *


warnings.warn(
    "The basekernel module has been renamed to graphdot.microkernel.\n"
    "Please update relevant import statements from\n"
    "'from graphdot.kernel.marginalized.basekernel import XXX'\n"
    "to\n"
    "'from graphdot.microkernel import XXX'",
    DeprecationWarning
)
