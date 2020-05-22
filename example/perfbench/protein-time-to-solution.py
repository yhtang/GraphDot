#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Performance benchmark on molecular graphs generated from protein crystal
structures."""
import json
import sys
import os
import re
from ase import Atoms
from graphdot.graph import Graph
from graphdot.kernel.molecular import Tang2019MolecularKernel

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

file = arg_dict.pop('file', 'pdb-3kDa-1324.json')
active = json.loads('[' + arg_dict.pop('active', 1) + ']')
zoom = arg_dict.pop('zoom', 1.5)
# reorder = arg_dict['reorder'] if 'reorder' in arg_dict else 'natural'

sys.stderr.write('Loading file %s\n' % file)
try:
    pdb_json = json.loads(open(file).read())
except FileNotFoundError:
    pdb_json = json.loads(
        open(os.path.join(os.path.dirname(__file__), file)).read()
    )
graph_list = []

for i in active:

    mol = pdb_json[i]

    sys.stderr.write(
        '%5d: %s, %d atoms\n' % (i, mol['pdb_id'], len(mol['sym']))
    )

    atoms = Atoms(mol['sym'], mol['xyz'])

    graph_list.append(Graph.from_ase(atoms, adjacency=dict(h=zoom)))

kernel = Tang2019MolecularKernel()

print(kernel(graph_list))
