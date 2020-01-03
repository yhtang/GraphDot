#!/usr/bin/env python
# -*- coding: utf-8 -*-
from ._backend import Backend
from ._backend_cuda import CUDABackend


def backend_factory(backend, *args, **kwargs):
    if isinstance(backend, Backend):
        return backend
    elif backend == 'cuda':
        return CUDABackend(*args, **kwargs)
    elif backend == 'auto':
        try:
            return CUDABackend(*args, **kwargs)
        except Exception as e:
            raise RuntimeError(f'Cannot auto-select backend: {e}')
    else:
        raise ValueError(f'Unknown backend {backend}')
