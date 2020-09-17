.. GraphDot documentation master file, created by
   sphinx-quickstart on Mon Jul 15 23:21:24 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to GraphDot's documentation!
====================================

.. image:: https://gitlab.com/yhtang/graphdot/badges/master/pipeline.svg
   :target: https://gitlab.com/yhtang/graphdot/commits/master

.. image:: https://gitlab.com/yhtang/graphdot/badges/master/coverage.svg
   :target: https://gitlab.com/yhtang/graphdot/commits/master

.. image:: https://img.shields.io/badge/License-BSD%203--Clause-blue.svg
   :TARGET: https://opensource.org/licenses/BSD-3-Clause

.. image:: https://badge.fury.io/py/graphdot.svg
   :target: https://badge.fury.io/py/graphdot

.. image:: https://readthedocs.org/projects/graphdot/badge/?version=latest&style=flat
   :target: https://graphdot.readthedocs.org/

GraphDot is a GPU-accelerated Python library that carries out graph dot product operations to compute graph similarity. Currently, the library implements the Marginalized Graph Kernel algorithm, which uses a random walk process to compare subtree patterns and thus defining a generalized graph convolution process. The library can operate on undirected graphs, either weighted or unweighted, that contain arbitrary nodal and edge labels and attributes. It implements state-of-the-art GPU acceleration algorithms and supports versatile customization through just-in-time code generation and compilation.


Features
--------
- Compares graph with different number of nodes and/or edges.
- Allows user to define arbitrary attributes and custom similarity functions on individual nodes and edges.
- Fast, memory-efficient GPU algorithms for CUDA.
- Compatible with major graph libraries such as NetworkX and graphviz.
- Interoperable with scikit-learn.
- Built-in specialization for chemistry and material science applications.

Contents
--------
.. toctree::
   :maxdepth: 2

   installation
   example
   quickstart
   marginalized
   api
   contribute
   changelog


Citation
--------

Like the package? Please cite:

- Tang, Yu-Hang, and Wibe A. de Jong. "Prediction of atomization energy using graph kernel and active learning." The Journal of chemical physics 150, no. 4 (2019): 044107.
- Tang, Yu-Hang, Oguz Selvitopi, Doru Thom Popovici, and Aydın Buluç. "A High-Throughput Solver for Marginalized Graph Kernels on GPU." In 2020 IEEE International Parallel and Distributed Processing Symposium (IPDPS), pp. 728-738. IEEE, 2020.

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


Contributors
------------

Yu-Hang "Maxin" Tang, Oguz Selvitopi, Doru Popovici, Yin-Jia Zhang

Copyright
---------

GraphDot Copyright (c) 2019, The Regents of the University of California,
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

Funding Acknowledgment
--------------------------------------------------------------------------------

This work was supported by the Luis W. Alvarez Postdoctoral Fellowship at Lawrence Berkeley National Laboratory. This work is also supported in part by the Applied Mathematics program of the DOE Office of Advanced Scientific Computing Research under Contract No. DE-AC02-05CH11231, and in part by the Exascale Computing Project (17-SC-20-SC), a collaborative effort of the U.S. Department of Energy Office of Science and the National Nuclear Security Administration.
