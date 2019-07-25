3D Molecular Graph
--------------------------------------------------------------------------------

.. literalinclude:: ../../example/molecular-3d.py
   :language: python
   :linenos:

Exptected output:

.. code-block:: none

                     H2O       HCl      NaCl  NaCl-bulk  NaCl-bulk2
    H2O         1.000000  0.073903  0.031434   0.031434    0.031434
    HCl         0.073903  1.000000  0.015842   0.015842    0.015841
    NaCl        0.031434  0.015842  1.000000   0.023764    0.023764
    NaCl-bulk   0.031434  0.015842  0.023764   1.000000    0.803760
    NaCl-bulk2  0.031434  0.015841  0.023764   0.803760    1.000000
