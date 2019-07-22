Installation
============

Prerequisites
--------------------------------------------------------------------------------

GraphDot requires a CUDA Toolkit installation to carry out GPU computations.
To install it, see https://developer.nvidia.com/cuda-toolkit.


Installation using pip
--------------------------------------------------------------------------------

For core functionality (GPU-accelerated marginalized graph kernel) only:

.. code-block:: bash

    pip install graphdot

For molecular modeling and interoperability with the ASE and pymatgen package:

.. code-block:: bash

    pip install graphdot[molecular]


Installation from source
--------------------------------------------------------------------------------

Ubuntu/Fedora/macOS
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

.. code-block:: bash

    git clone https://gitlab.com/yhtang/graphdot
    cd graphdot
    pip3 install -r requirements/common.txt
    python3 setup.py install
