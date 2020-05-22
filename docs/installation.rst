Installation
============

Prerequisites
--------------------------------------------------------------------------------

GraphDot requires a CUDA Toolkit installation for carrying out GPU computations.
To install it, following the instructions on
https://developer.nvidia.com/cuda-toolkit.


Installation using pip
--------------------------------------------------------------------------------

GraphDot can be installed from PyPI as simple as:

.. code-block:: bash

    pip install graphdot


Install from source
--------------------------------------------------------------------------------

For Ubuntu/Fedora/macOS
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

.. code-block:: bash

    git clone https://gitlab.com/yhtang/graphdot
    cd graphdot
    pip3 install -r requirements/common.txt
    python3 setup.py install
