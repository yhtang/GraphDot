#!/usr/bin/env python
# -*- coding: utf-8 -*-
from .gpr import GaussianProcessRegressor
from .nystrom import LowRankApproximateGPR
from .outlier_detector import GPROutlierDetector

__all__ = [
    'GaussianProcessRegressor', 'LowRankApproximateGPR', 'GPROutlierDetector'
]
