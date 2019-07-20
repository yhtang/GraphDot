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

.. image:: https://readthedocs.org/projects/pip/badge/?version=latest&style=flat
   :target: https://graphdot.readthedocs.org/

GraphDot is a library for similarity comparison between labeled and weighted graphs. Currently, it implements the Marginalized Graph Kernel algorithm. It is GPU-accelerated and supports versatile customization through just-in-time code generation and compilation.

Features
--------
- Compares graph with different number of nodes and/or edges.
- Allows user to define arbitrary attributes and custom similarity functions on individual nodes and edges.
- Fast, memory-efficient GPU algorithms for CUDA.
- Compatible with major graph libraries such as NetworkX and graphviz.
- Interoperable with scikit-learn.
- Built-in specialization for chemistry and material science applications.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   example
   userguide
   api


Citation
--------

Like the package? Please cite:

Tang, Y. H., & de Jong, W. A. (2019). Prediction of atomization energy using graph kernel and active learning. *The Journal of chemical physics*, 150(4), 044107. https://doi.org/10.1063/1.5078640


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


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
