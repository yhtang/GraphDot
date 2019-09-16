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
- Switched to single precision floating points for edge length in `Graph.from_ase`
- Added several performance benchmark code to `example/perfbench`