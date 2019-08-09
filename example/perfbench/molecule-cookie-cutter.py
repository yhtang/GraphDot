#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Performance benchmark on molecular graphs generated from small organic
molecules."""
import sys
import re
import numpy as np
import pycuda
from ase.build import molecule
from graphdot.graph import Graph
from graphdot.kernel.molecular import Tang2019MolecularKernel as Kernel


import pycuda.driver
import pycuda.gpuarray
from pycuda.gpuarray import GPUArray
from graphdot.kernel.marginalized import Job


arg_dict = {}
for arg in sys.argv[1:]:
    m = re.fullmatch(r"-([\w\d]+)=(.+)", arg)
    if m:
        try:
            arg_dict[m.group(1)] = int(m.group(2))
        except ValueError:
            try:
                arg_dict[m.group(1)] = float(m.group(2))
            except ValueError:
                arg_dict[m.group(1)] = m.group(2)
    else:
        sys.stderr.write('Unrecognized argument: %s\n' % arg)
        sys.exit(1)

print(arg_dict)

formula = arg_dict.pop('formula', 'CH3COOH')
zoom = arg_dict.pop('zoom', 1.5)
repeat = arg_dict.pop('repeat', 4)
nlaunch = arg_dict.pop('nlaunch', 1)
block_per_sm = arg_dict.pop('block_per_sm', 8)
block_size = arg_dict.pop('block_size', 128)
device = pycuda.driver.Device(arg_dict.pop('device', 0))

njobs = device.MULTIPROCESSOR_COUNT * block_per_sm * repeat

g = Graph.from_ase(molecule(formula), adjacency=dict(h=zoom))

kernel = Kernel()

''' generate jobs '''
jobs = [Job(0, 0, GPUArray(len(g.nodes)**2, np.float32))
        for i in range(njobs)]

''' call GPU kernel '''
for i in range(nlaunch):
    kernel.kernel._launch_kernel([g], jobs, nodal=False, lmin=0)

R = jobs[0].vr_gpu.get().reshape(len(g.nodes), -1)
r = R.sum()

print('Nodal similarity:\n', R, sep='')
print('Overall similarity:\n', r, sep='')

for job in jobs:
    assert(np.abs(job.vr_gpu.get().sum() - r) < r * 1e-6)
print('**ALL PASSED**')
