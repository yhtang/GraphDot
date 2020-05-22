#ifndef GRAPHDOT_BASEKERNEL_CONVOLUTION_H_
#define GRAPHDOT_BASEKERNEL_CONVOLUTION_H_

#include <fmath.h>

namespace graphdot {

namespace basekernel {

template<class F, class X, class Y>
inline __host__ __device__ float convolution(
    F const f,
    X const x,
    Y const y
) {
    float kxx = 0, kxy = 0, kyy=0;
    for(auto const &_1: x) {
        for(auto const &_2: x) {
            kxx += f(_1, _2);
        }
    }
    for(auto const &_1: x) {
        for(auto const &_2: y) {
            kxy += f(_1, _2);
        }
    }
    for(auto const &_1: y) {
        for(auto const &_2: y) {
            kyy += f(_1, _2);
        }
    }
    if (kxx > 0 && kyy > 0) {
        return kxy * rsqrtf(kxx * kyy);
    } else {
        return 0.f;
    }
}

template<class F, class J, class X, class Y>
inline __host__ __device__ float convolution_jacobian(
    F const f,
    J const j,
    X const x,
    Y const y
) {
    float kxx = 0, kxy = 0, kyy=0, jxx = 0, jxy = 0, jyy = 0;
    for(auto const &_1: x) {
        for(auto const &_2: x) {
            kxx += f(_1, _2);
            jxx += j(_1, _2);
        }
    }
    for(auto const &_1: x) {
        for(auto const &_2: y) {
            kxy += f(_1, _2);
            jxy += j(_1, _2);
        }
    }
    for(auto const &_1: y) {
        for(auto const &_2: y) {
            kyy += f(_1, _2);
            jyy += j(_1, _2);
        }
    }
    if (kxx > 0 && kyy > 0) {
        auto kxx_kyy_3 = ipow<3>(kxx * kyy);
        return jxy * rsqrtf(kxx * kyy) - 0.5f * kxy * rsqrtf(kxx_kyy_3) * (jxx * kyy + kxx * jyy);
    } else {
        return 0.f;
    }
}

}

}

#endif