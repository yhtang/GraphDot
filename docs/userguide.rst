.. _user_guide:

User Guide
==========

Marginalized Graph Kernel
-------------------------

In the framework of the marginalized graph kernel algorithm [Kashima, H., Tsuda, K., & Inokuchi, A. (2003). Marginalized kernels between labeled graphs. *In Proceedings of the 20th international conference on machine learning (ICML-03)* (pp. 321-328).], the similarity between two graphs is computed as the expectation of individual similarities of pairs of paths generated from a random walk process. Thus, to carry out the computation, users need to define a node similarity kernelet and an edge similarity kernelet using the base kernels as defined below:

.. autofunction:: graphdot.kernel.marginalized.basekernel.Constant
   :noindex:
.. autofunction:: graphdot.kernel.marginalized.basekernel.KroneckerDelta
   :noindex:
.. autofunction:: graphdot.kernel.marginalized.basekernel.SquareExponential
   :noindex:
.. autofunction:: graphdot.kernel.marginalized.basekernel.TensorProduct
   :noindex:
