from graphdot.marginalized.basekernel import Constant
import struct
from graphdot.interop.structure import flatten
import pycuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import graphdot.cpp


k1 = Constant(42)
print(k1('a', 1))

k = k1

print(struct.pack(flatten(k1._layout)[0], *k1.theta))

cpu_mem = struct.pack(flatten(k1._layout)[0], *k1.theta)

gpu_mem = pycuda.driver.to_device(cpu_mem)

print(gpu_mem)

print(graphdot.cpp.__path__)

mod = SourceModule(r"""
#include <marginalized/basekernel.h>

extern "C" {
__global__ void test(%s const &x)
{
    printf("Hello! %f\n", x.constant);
    printf("World! %f\n", x(1, 2));
}
}""" % k._decltype, options=['-std=c++14'],
      no_extern_c=True,
      include_dirs=graphdot.cpp.__path__)



gpu_kernel = mod.get_function("test")
gpu_kernel(gpu_mem, block=(4, 1, 1), grid=(1, 1, 1))
