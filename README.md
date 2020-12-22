# The GraphDot Library

[![pipeline status](https://gitlab.com/yhtang/graphdot/badges/master/pipeline.svg)](https://gitlab.com/yhtang/graphdot/commits/master)
[![coverage report](https://gitlab.com/yhtang/graphdot/badges/master/coverage.svg)](https://gitlab.com/yhtang/graphdot/commits/master)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![PyPI version](https://badge.fury.io/py/graphdot.svg)](https://badge.fury.io/py/graphdot)
[![docs](https://readthedocs.org/projects/graphdot/badge/?version=latest&style=flat)](https://graphdot.readthedocs.org/)

GraphDot is a GPU-accelerated Python library that carries out graph dot product operations to compute graph similarity. Currently, the library implements the Marginalized Graph Kernel algorithm, which uses a random walk process to compare subtree patterns and thus defining a generalized graph convolution process. The library can operate on undirected graphs, either weighted or unweighted, that contain arbitrary nodal and edge labels and attributes. It implements state-of-the-art GPU acceleration algorithms and supports versatile customization through just-in-time code generation and compilation.

For more details, please checkout the latest documentation on [readthedocs](https://graphdot.readthedocs.io/).

# Copyright

GraphDot Copyright (c) 2019-2020, The Regents of the University of California,
through Lawrence Berkeley National Laboratory (subject to receipt of any
required approvals from the U.S. Dept. of Energy).  All rights reserved.

If you have questions about your rights to use or distribute this software,
please contact Berkeley Lab's Intellectual Property Office at
IPO@lbl.gov.

NOTICE.  This Software was developed under funding from the U.S. Department
of Energy and the U.S. Government consequently retains certain rights.  As
such, the U.S. Government has been granted for itself and others acting on
its behalf a paid-up, nonexclusive, irrevocable, worldwide license in the
Software to reproduce, distribute copies to the public, prepare derivative
works, and perform publicly and display publicly, and to permit other to do
so.

# Like the package?

Please cite:

- Tang, Yu-Hang, and Wibe A. de Jong. "Prediction of atomization energy using graph kernel and active learning." The Journal of chemical physics 150, no. 4 (2019): 044107.
- Tang, Yu-Hang, Oguz Selvitopi, Doru Thom Popovici, and Aydın Buluç. "A High-Throughput Solver for Marginalized Graph Kernels on GPU." In 2020 IEEE International Parallel and Distributed Processing Symposium (IPDPS), pp. 728-738. IEEE, 2020.
