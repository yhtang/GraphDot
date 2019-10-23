#!/usr/bin/env python
# -*- coding: utf-8 -*-


def pytest_benchmark_scale_unit(config, unit, benchmarks, best, worst, sort):
    if unit == 'seconds':
        prefix = 'm'
        scale = 1e3
    elif unit == 'operations':
        prefix = 'K'
        scale = 0.001
    else:
        raise RuntimeError("Unexpected measurement unit %r" % unit)

    return prefix, scale
