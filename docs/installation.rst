Installation
============

Prerequisites
--------------------------------------------------------------------------------

GraphDot requires a CUDA Toolkit installation to carry out GPU computations. To install it, see https://developer.nvidia.com/cuda-toolkit.


Installation using pip
--------------------------------------------------------------------------------

.. code-block:: bash

    pip install graphdot


Installation from source
--------------------------------------------------------------------------------

Ubuntu
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

.. code-block:: bash

    apt update
    apt install python3-pip
    git clone https://gitlab.com/yhtang/graphdot
    cd graphdot
    pip3 install -r requirements.txt
    python3 setup.py install
