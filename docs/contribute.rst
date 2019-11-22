How to contribute
=================

GraphDot is an open-source project released under a BSD license.
Everyone is welcome to contribute.

The project is hosted on `GitLab <https://gitlab.com/yhtang/graphdot>`_.
For questions, suggestions and bug reports, please take advantage of the
`issue tracking system <https://gitlab.com/yhtang/graphdot/issues>`_.
In addition, contributions are very welcomed and could be submitted as
`merge requests <https://gitlab.com/yhtang/graphdot/merge_requests>`_.


Submitting a bug report or a feature request
++++++++++++++++++++++++++++++++++++++++++++

Please feel free to open an issue should you run into a bug or wish a feature
could be implemented.

When submitting an issue, please try to follow the guidelines below:

- Include a minimal reproducible example of the issue for bug reports.
- Provide a mock code snippt for feature suggestions.
- Provide a full traceback when an exception is raised.
- Please include your operating system type and version number, as well as your
  Python, graphdot, numpy, and pycuda versions. This information can be found
  by running:

  .. code-block:: Python

     import platform; print(platform.platform())
     import sys; print('Python', sys.version)
     import graphdot; print('GraphDot', graphdot.__version__)
     import numpy; print('NumPy', numpy.__version__)
     import pycuda; print('PyCUDA', pycuda.VERSION)


Contributing Code
+++++++++++++++++

The most recommended way to contribute to GraphDot is to fork the
`main repository <https://gitlab.com/yhtang/graphdot>`_, then submit a
"merge request" following the procedure below:

1. `Fork <https://docs.gitlab.com/ee/gitlab-basics/fork-project.html>`_ the project repository.
2. Clone your own fork to local disk via ``git clone``
3. `Setting up the development environment`_
4. Create a branch for development via ``git checkout -b feature/<feature-name> master`` (replace ``feature-name`` with the actual name of the feature).
5. Make changes on the feature branch
6. Test the changes with `Quality assurance measures`_.
7. Push the completed feature to your own fork, then
   `create a merge request <https://docs.gitlab.com/ee/gitlab-basics/add-merge-request.html>`_.


Development Guide
+++++++++++++++++

Setting up the development environment
--------------------------------------

A recipe is provided in the project's Makefile for creating a Python virtual
environment containing all dependencies for development:

.. code-block:: bash

   make setup


Alternatively, a virtual environment can be set up manually by:

.. code-block:: bash

   virtualenv venv
   source venv/bin/activate
   pip install -r requirements/common.txt
   pip install -r requirements/tests.txt
   pip install -r requirements/docs.txt


Python code style
-----------------

Variable, function, file, module, and package names should use all lower case
letters with underscores being word separators. For example, use ``module_name.py``
instead of ``Module-Name.py``, and ``some_function()`` instead of ``SomeFunction()``.
Class names should use the Pascal case.
For example, use ``ClassName`` instead of ``class_name``.


In addition, the `PEP8 <https://www.python.org/dev/peps/pep-0008/>`_ style guide
should be followed. An (incomplete) summary of the style guide is:

- 4 spaces per indentation level
- 79 characters at most per line
- 2 blank lines around top-level functions and class definitions
- 1 blank line around method definitions inside a class
- 1 module per import line, imports always at top of file
- UTF-8 source file encoding
- 0 space on the inner side of parentheses, e.g. use ``(x + y)`` instead of ``( x + y )``.
- 1 space on both sides of binary operators (including assignments), e.g. use ``x = y * 3 + z`` instead of ``x=y*3+z``.
- 0 space around keyword argument assignments, e.g. use ``f(x=1)`` instead of ``f(x = 1)``.

Comformance to the style guide can be checked via `Code style check`_.

C++ code style
--------------

A configuration file for `clang-format <https://clang.llvm.org/docs/ClangFormatStyleOptions.html>`_
has been defined in the root directory of the repository. It can be used to format
C++ files using:

.. code-block:: bash

   clang-format -i files



Quality assurance measures
--------------------------


Unit tests
**********

.. code-block:: bash

   make test

Or alternatively

.. code-block:: bash

   tox -e py37  # or py35, py36 etc.


Code style check
****************

.. code-block:: bash

   make lint


Coverage test
*************

.. code-block:: bash

   make test-coverage

Or alternatively

.. code-block:: bash

   tox -e coverage

Coverage reports are stored in the ``htmlcov`` directory.


Performance Benchmark
*********************

.. code-block:: bash

   tox -e benchmark
