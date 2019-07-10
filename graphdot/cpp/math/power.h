#ifndef GRAPHDOT_MATH_POWER_H
#define GRAPHDOT_MATH_POWER_H

namespace graphdot {

template<class T> constexpr inline T power(T x, unsigned int y) noexcept {
    return y ? x * power(x, y - 1) : 1;
}

}

#endif
