#ifndef GRAPHDOT_BASEKERNEL_NORMALIZE_H_
#define GRAPHDOT_BASEKERNEL_NORMALIZE_H_

#include <fmath.h>

namespace graphdot {

namespace basekernel {

template<class F, class X, class Y>
inline __host__ __device__ float normalize(
    F const f,
    X const x,
    Y const y
) {
    auto const kxx = f(x, x);
    auto const kyy = f(y, y);
    if (kxx * kyy > 0) {
        return f(x, y) * rsqrtf(kxx * kyy);
    } else {
        return 0.f;
    }
}

template<class F, class J, class X, class Y>
inline __host__ __device__ float normalize_jacobian(
    F const f,
    J const j,
    X const x,
    Y const y
) {
    auto const kxx = f(x, x);
    auto const kxy = f(x, y);
    auto const kyy = f(y, y);
    auto const jxx = j(x, x);
    auto const jxy = j(x, y);
    auto const jyy = j(y, y);

    if (kxx * kyy > 0) {
        auto const kxx_kyy_3 = ipow<3>(kxx * kyy);
        return jxy * rsqrtf(kxx * kyy) - 0.5f * kxy * rsqrtf(kxx_kyy_3) * (jxx * kyy + kxx * jyy);
    } else {
        return 0.f;
    }
}

}

}

#endif