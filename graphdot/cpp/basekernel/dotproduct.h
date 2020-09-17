#ifndef GRAPHDOT_BASEKERNEL_DOTPRODUCT_H_
#define GRAPHDOT_BASEKERNEL_DOTPRODUCT_H_

#include <fmath.h>
#include <frozen_array.h>

namespace graphdot {

namespace basekernel {

template<class T>
inline __host__ __device__ float dotproduct(
    graphdot::numpy_type::frozen_array<T> const x,
    graphdot::numpy_type::frozen_array<T> const y
) {
    T sum = 0;
    for(int i = 0; i < x.size; ++i) {
        sum += x._data[i] * y._data[i];
    }
    return sum;
}

}

}

#endif