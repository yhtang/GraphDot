#ifndef GRAPHDOT_BASEKERNEL_CONVOLUTION_H_
#define GRAPHDOT_BASEKERNEL_CONVOLUTION_H_

#include <fmath.h>

namespace graphdot {

namespace basekernel {

template<bool mean, class F, class X, class Y>
inline __host__ __device__ float convolution(
    F const f,
    X const x,
    Y const y
) {
    float k = 0;
    for(auto const &_1: x) {
        for(auto const &_2: y) {
            k += f(_1, _2);
        }
    }
    if (mean) {
        return k / (x.size * y.size);
    } else {
        return k;
    }
}

template<bool mean, class J, class X, class Y>
inline __host__ __device__ float convolution_jacobian(
    J const j,
    X const x,
    Y const y
) {
    float dk = 0;
    for(auto const &_1: x) {
        for(auto const &_2: y) {
            dk += j(_1, _2);
        }
    }
    if (mean) {
        return dk / (x.size * y.size);
    } else {
        return dk;
    }
}

}

}

#endif