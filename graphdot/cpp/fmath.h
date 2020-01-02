#ifndef GRAPHDOT_FMATH_H_
#define GRAPHDOT_FMATH_H_

namespace graphdot {

namespace detail {

template<class F, int E> struct _ipow {
    constexpr static inline F compute(F base) {
        F a = _ipow<F, E / 2>::compute(base);
        F b = _ipow<F, E % 2>::compute(base);
        return a * a * b;
    }
};

template<class F> struct _ipow<F, 0> {
    constexpr static inline F compute(F base) { return F(1); }
};

template<class F> struct _ipow<F, 1> {
    constexpr static inline F compute(F base) { return base; }
};

}  // namespace detail

template<int E, class F> constexpr inline F ipow(F base) {
    return detail::_ipow<F, E>::compute(base);
}

template<int E, class F> constexpr inline F ripow(F base) {
    return detail::_ipow<F, E>::compute(1 / base);
}

}  // namespace graphdot

#endif
