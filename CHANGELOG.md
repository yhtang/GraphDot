# Change log of GraphDot

## 0.5a1 (2020-06-17)

- Added a built-in Gaussian process regression (GPR) module.
- Fixed an issue that prevented the pickling of graphs.

## 0.4.6 (2020-06-05)

- Fixed a minor bug in `Graph.from_rdkit`.

## 0.4.5 (2020-05-26)

- Replaced `from_smiles` with a more robust `from_rdkit` function with
  additional ring stereochemistry features. Thanks to [Yan Xiang](mailto:hnxxxy123@sjtu.edu.cn) for the contribution.
- Added a new `Compose` method for creating base kernels beyond tensor product
  base kernels.
- Fixed a performance degradation issue (#57).


## 0.4.4 (2020-05-23)

- Ensure that graphs can be pickled.

## 0.4.3 (2020-05-23)

- Ensure graph feature data layout consistency involving a mixture of scalar
  and variable-length features. Fixes #56.

## 0.4.2

- Fixed an integer sign issue introduced with graph type unification.

## 0.4.1

- Renamed `Graph.normalize_types` to `Graph.unify_datatype`.

## 0.4.0

- Now allowing variable-length node and edge features thanks to a redesign of
  the Python/C++ data interoperation mechanism.
- Introduced a `Convolution` base kernel for composing kernels on
  variable-length attributes using scalar base kernels.

## 0.3.5

- Added a `dtype` option to the `MarginalizedGraphKernel` to specify the type of returned matrix elements.

## 0.3.4

- Specified the minimum version of sympy in installation requirements.

## 0.3.3

- Allow M3 metric to use partial charge information.
- Made the element, bond, and charge parameters adjustable in the M3 metric.

## 0.3.2

- Miscellaneous bug fixes.

## 0.3.1

- Analytic computation of graph kernel derivatives against hyperparameters.

## 0.3.0

- Users can now define new base kernels easily using SymPy expression #45.
- Better scikit-learn interoperability.

## 0.2.9 (2019-12-14)

- Fixed a bug related to atomic adjacency #43.

## 0.2.8 (2019-11-22)

- Added an experimental 'M3' distance metric

## 0.2.7 (2019-11-18)

- Bug fixes and stability improvements

## 0.2.6 (2019-10-31)

- Improved the performance of hyperparameter optimization by enabling lightweight re-parameterization.
- Implemented a few properties and methods for scikit-learn interoperability.

## 0.2.5 (2019-10-30)

- Improved the performance of successive graph kernel evaluations

## 0.2.4 (2019-10-29)

- Improved the performance of graph format conversion for the GPU kernel by 3 times.

## 0.2.3 (2019-10-24)

- Incorporated many new optimizations as detailed in https://arxiv.org/abs/1910.06310.
- Preparing for faster memory allocation and job creation.
- Fixes #32, #33.

## 0.2.1 (2019-10-02)

- Reduced kernel launch preparation time by 50% to address #28.
- Fixed a memory leak issue #31.

## 0.2.0 (2019-09-26)

- Changed return type of the `diag()` method of `MarginalizedGraphKernel` to fix #30.

## 0.1.9 (2019-09-25)

- Fixed an edge label consistency issue with graphs generated from SMILES strings.

## 0.1.8 (2019-09-15)

- Added a freshly-designed atomic adjacency rule.
- Significantly accelerated conversion from ASE molecules to graphs.
- Documentation update.

## 0.1.7 (2019-08-23)

- Documentation update.

## 0.1.6 (2019-08-12)

- Added the `diag()` method to `Tang2019MolecularKernel`.

## 0.1.5 (2019-08-09)

- Fixed a regression in the CUDA kernel that caused an order-of-magnitude slowdown
- Switched to single-precision floating points for edge length in `Graph.from_ase`
- Added several performance benchmark code to `example/perfbench`
