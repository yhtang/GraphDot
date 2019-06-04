import graphdot
import cppyy

__all__ = [ 'TensorProduct', 'Convolution', 'Unity', 'KroneckerDelta', 'SquareExponential' ]

TensorProduct     = cppyy.gbl.graphdot.kernel.make_tensor_product_kernel
Convolution       = cppyy.gbl.graphdot.kernel.make_convolutional_kernel
Unity             = cppyy.gbl.graphdot.kernel.unity
KroneckerDelta    = cppyy.gbl.graphdot.kernel.kronecker_delta
SquareExponential = cppyy.gbl.graphdot.kernel.square_exponential
