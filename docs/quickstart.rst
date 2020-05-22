Quick Start Tutorial
====================

Marginalized Graph kernel
-------------------------

This quick-start guide here assumes that the reader is already familiar with
the marginalized graph kernel algorithm [Kashima, H., Tsuda, K., & Inokuchi,
A. (2003). Marginalized kernels between labeled graphs. *In Proceedings of the
20th international conference on machine learning (ICML-03)* (pp. 321-328).].
Otherwise, please refer to :ref:`tutorial_on_marginalized_graph_kernel`.

Graphs can be imported from a variety of formats, such as
`a networkx undirected Graph <https://networkx.github.io/documentation/stable/reference/classes/graph.html>`_, 
`an ASE atoms collection <https://wiki.fysik.dtu.dk/ase/ase/atoms.html>`_,
`a pymatgen structure or molecule <https://pymatgen.org/pymatgen.core.structure.html>`_,
`a SMILES string <https://en.wikipedia.org/wiki/Simplified_molecular-input_line-entry_system>`_.

To compare the overall similarity between two graphs, the user needs to supply
two base `kernels <https://www.cs.toronto.edu/~duvenaud/cookbook/>`_:
one for comparing individual nodes, and another one for comparing individual
edges. The base kernels can be picked from a library of prebuilt kernels as
defined in the graphdot.kernel.basekernel module:

.. autofunction:: graphdot.kernel.basekernel.Constant
   :noindex:
.. autofunction:: graphdot.kernel.basekernel.KroneckerDelta
   :noindex:
.. autofunction:: graphdot.kernel.basekernel.SquareExponential
   :noindex:
.. autofunction:: graphdot.kernel.basekernel.TensorProduct
   :noindex:
