# Change log of GraphDot

## 0.8.1 (2021-12-08)

- Hotfix to improve GFR numerical stability when adjacency is nearly zero.

## 0.8 (2021-12-07)

This version formalizes the inclusion of new features introduced from 0.8a1
to 0.8a18. An (incomplete) list of features include:
- Dataset downloaders (`graphdot.dataset`)
- Graph Hausdorff distance metric (`graphdot.metric.maximin`)
- Gaussian field regressor (`graphdot.model.gfr`)
- Kernel-induced distance metrics (`graphdot.metric`)
- Low-rank GPR via Nystrom approximation (`graphdot.model.gpr.nystrom`)
- Multiplicative regularization for GPR

## 0.8a18 (2021-09-30)

- Maintenance update.

## 0.8a17 (2021-03-12)

- Fixed a QM9 downloader issue.
- Fixed a bug with the Maximin metric when some hyperparameters are fixed.

## 0.8a15 (2021-03-12)

- Downloader for QM9.

## 0.8a14 (2021-03-09)

- Minor tweaks to look-ahead rewriter logic.

## 0.8a13 (2021-03-08)

- A new and experimental calling convention to allow evaluations of the
  graph kernels at a list of specific indices.

## 0.8a12 (2021-03-07)

- A sequence-based rewriter for Monte Carlo tree search.
- Convert any kernel into a metric via `KernelInducedDistance`.
- Convert any norm into a kernel via `KernelOverMetric`.

## 0.8a11 (2021-02-25)

- Improvements to the active learning hierarchical drafter.

## 0.8a10 (2021-01-14)

- Performance optimization for GFR leave-one-out cross validation gradients.

## 0.8a9 (2021-01-12)

- Leave-one-out cross validation for Gaussian field regressor.

## 0.8a8 (2021-01-05)

- Normalized the MaxiMin graph distance metric to [0, 1].
- More data downloaders: METLIN SMRT, AMES, and a custom downloader.

## 0.8a7 (2021-01-02)

- Gradient evaluation for the MaxiMin graph metric.
- Gradient evaluation for Gaussian field regressor prediction loss.

## 0.8a6 (2020-12-29)

- Optimized the evaluation of the gradient of the loss function for the
  Gaussian field regressor.
- Implemented a finite-difference based graph kernel nodal gradient.

## 0.8a5 (2020-12-21)

- Added a downloader for the QM7 dataset.
- Prototype implementation of a Gaussian field harmonic function regressor.

## 0.8a4 (2020-12-15)

- Added an multiplicative regularization option to GPR, which may perform
  better when the kernel is not normalized.
- Fixed a linear algebra type error when the GPR kernel matrix is solved
  with pseudoinverse.

## 0.8a3 (2020-11-23)

- Added an experimental Monte Carlo tree search model.

## 0.8a2 (2020-11-16)

- Enabled Low-rank GPR (Nystrom) training with missing target values.

## 0.8a1 (2020-10-15)

- Enabled GPR training with missing target values.

## 0.7 (2020-09-21)

This version formalizes the inclusion of new features introduced from 0.7a1
to 0.7b2. An (incomplete) list of features include:
- A redesigned active learning module (`graphdot.model.active_learning`).
- The PBR graph reordering algorithm for graph kernel acceleration
  (`graphdot.graph.reorder.pbr`).
- LOOCV predictions using the low-rank approximate GPR.
- Significant improvement to the robustness of the training methods of GPR
  and Low-rank GPR models.
- Allow kernel/microkernel hyperparameters to be declared as 'fixed' via the
  `*_bounds` arguments.
- Added a `DotProduct` microkernel for vector-valued node and edge features.
- Added a `.normalized` attribute to all elementary and composite microkernels.
- Graph representation string can now be directly deserialized using `eval`.
- New atomic adjacency options such as alternative bell-shaped compact
  adjacency functions (`compactbell[a,b]`), and new length scale choices using covalent radiu etc.
- Perform value range check for the node and edge kernels during graph
  kernel creation.
- Added a `to_networkx()` method to `graphdot.Graph`.
- Enhanced the readability of the string representations of kernel
  hyperparameters using an indented print layout.
- Various performance and bug fixes.

## 0.7b2 (2020-09-16)

- Added a `DotProduct` microkernel for vector-valued node and edge features.
- Added a `.normalized` attribute to all elementary and composite microkernels.
- Perform value range check for the node and edge kernels during graph kernel
  creation.

## 0.7b1 (2020-09-12)

- Performance improvements to the variance minimizing active learner.

## 0.7a13 (2020-09-10)

- Furture improvements to the robustness of the GPR training process.

## 0.7a12 (2020-09-02)

- Uses a more robust pseudoinverse algorithm for GPR when the kernel matrix
  is nearly singular.

## 0.7a11 (2020-09-02)

- Added bell-shaped compact adjacency functions.
- Redesigned the active learning module.

## 0.7a10 (2020-08-30)

- Enhanced the readability of the string representations of kernel
  hyperparameters.
- New atomic adjacency options.

## 0.7a9 (2020-08-28)

- Improved numerical stability tolerance of the GPR and Low-rank GPR models.

## 0.7a8 (2020-08-27)

- Added a `to_networkx()` method to `graphdot.Graph`.

## 0.7a7 (2020-08-25)

- Graph representation string can now be directly deserialized using `eval`.

## 0.7a6 (2020-08-23)

- Optimized GPU gradient evaluation performance
- `predict_loocv` now available for the LowRankApproximateGPR model.
- Unified the `fit` and `fit_loocv` method of GaussianProcessRegressor.

## 0.7a4 (2020-08-18)

- Fixed a bug related to bounds of kernels contains fixed hyperparameters.

## 0.7a3 (2020-08-18)

- Allow kernel/microkernel hyperparameters to be declared as 'fixed' via the
  `*_bounds` arguments.

## 0.7a2 (2020-08-14)

- Fixed a memory layout issue that slowed down computations using normalized
  kernels.

## 0.7a1 (2020-08-12)

- The PBR graph reordering algorithm as proposed in
  [10.1109/IPDPS47924.2020.00080][ipdps] is now available.

[ipdps]: https://ieeexplore.ieee.org/abstract/document/9139866

## 0.7a (2020-08-10)

- Improved the performance of gradient evaluation for the marginalized graph
  kernel.
- Introduced a new `MaxiMin` distance metric between graphs.

## 0.6.6 (2020-08-10)

- Added `save` and `load` methods to the Gaussian process regressor models.

## 0.6.5 (2020-08-05)

- Fixed a bug related to the `lmin=1` option of the marginalized graph kernel.

## 0.6.4 (2020-08-03)

- Fixed a bug regarding target value normalization in the `fit_loocv` method of
  GPR.

## 0.6.3 (2020-07-30)

- Fixed a performance degradation due to the inconsistent lexical sorting
behavior between `numpy.lexsort` and `numpy.unique`.

## 0.6.2 (2020-07-30)

- Fixed a bug in computing the gradient of diagonal kernel entries.

## 0.6.1 (2020-07-29)

- Fixed a bug in kernel normalization.

## 0.6 (2020-07-26)

This version formally releases the new features as have been introduced in
the various 0.6alpha versions, such as:
- Nystrom low-rank approximate Gaussian process regressor
- Graphs with self-looping edges
- Graph permutation and reordering operations for GPU performance boost.
- Hyperparameterized and optimizable starting probabilities for the graph
  kernel.

## 0.6a10 (2020-07-21)

- Supports graphs with self-looping edges.
- Made the `Graph.from_rdkit` method optional in case if RDKit itself is
  not available.

## 0.6a9 (2020-07-17)

- Ensures that graph cookies are not pickled.

## 0.6a7, 0.6a8 (2020-07-16)

- Fixed a problem assocaited with converting permuted graphs to octilegraphs.

## 0.6a6 (2020-07-16)

- Fixed a problem with caching graphs on the GPU.

## 0.6a5 (2020-07-15)

- Introduced a graph reordering mechanism to improve computational performance
  on GPUs.
- The default starting probability of the marginalized graph kernel is now
  hyperparameterized and will be optimized by default during training.
- Allow users to specify custom starting probability distributions.
- Performance improvements due to the in situ computation of starting
  probabilities instead of loading from memory.
- Added `repeat`, `theta_jitter` and `tol` options to the Gaussian process
  regressor.
- Fixed a normalization bug in `GaussianProcessRegressor.fit_loocv`.

## 0.5.1 (2020-06-30)

- Added a verbose training progress option to the GPR module. 
- The `graphdot.kernel.basekernel` package has been redesigned and renamed to
  `graphdot.microkernel`.

## 0.5 (2020-06-30)

- Initial formal release of the Gaussian Process regresion module.

## 0.5a7 (2020-06-28)

- Implemented the base kernel exponentiation, i.e. `k**a`, semantics.
- Minor docstring fixes.

## 0.5a6 (2020-06-26)

- Fixed a regression that causes data frame unpickling errors.

## 0.5a5 (2020-06-24)

- Added the leave-one-out cross-validation prediction and training to GPR.

## 0.5a4 (2020-06-22)

- Fixed an automatic documentation issue.

## 0.5a3 (2020-06-20)

- Added check for the shape of hyperparameter bounds specification to prevent
  users from unknowingly provide invalid values.

## 0.5a2 (2020-06-20)

- Fixed a bug related to Jacobian dimensionality.

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
