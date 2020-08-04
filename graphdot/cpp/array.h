#ifndef GRAPHDOT_ARRAY_H_
#define GRAPHDOT_ARRAY_H_

namespace graphdot {

// Gracefully handles 0-length arrays
template<class T, int size> struct array       {using type = T[size];};
template<class T>           struct array<T, 0> {using type = T *;};
template<class T, int n> using array_t = typename array<T, n>::type;

}

#endif
