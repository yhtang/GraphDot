#ifndef GRAPHDOT_MARGINALIZED_BASEKERNEL_H_
#define GRAPHDOT_MARGINALIZED_BASEKERNEL_H_

#include <utility>
#include <string>
#include <cmath>
#include <math/utility.h>

namespace graphdot {

namespace basekernel {

#ifndef __GENERIC__
#define __GENERIC__ __inline__ __host__ __device__
#endif

/*-----------------------------------------------------------------------------
  constant kernel
-----------------------------------------------------------------------------*/
struct constant {
    template<class X, class Y>
    __GENERIC__ auto operator() ( X const & x, Y const & y ) const {
        return constant;
    }
    float constant;
};

/*-----------------------------------------------------------------------------
  Kronecker delta/identity kernel
-----------------------------------------------------------------------------*/
struct kronecker_delta {
    template<class X, class Y>
    __GENERIC__ auto operator() ( X const & x, Y const & y ) const {
        return x == y ? hi : lo;
    }
    float lo, hi;
};

/*-----------------------------------------------------------------------------
  Gaussian kernel
-----------------------------------------------------------------------------*/
struct square_exponential {
    template<class X, class Y>
    __GENERIC__ auto operator() ( X const & x, Y const & y ) const {
        return expf( 0.5f * ( x - y ) * ( x - y ) / (length_scale * length_scale) );
    }
    float length_scale;
};

/*-----------------------------------------------------------------------------
  tensor product kernel
-----------------------------------------------------------------------------*/
template<class T> __GENERIC__ auto reduce( T value ) {
    return value;
}

template<class T, class ...Ts> __GENERIC__ auto reduce( T head, Ts ...tail ) {
    return head * reduce(tail...);
}

template<class Kernels, std::size_t ...idx> struct tensor_product {
    template<class X, class Y>
    __GENERIC__ auto operator() ( X && x, Y && y ) const {
        return reduce( std::get<idx>( kernels )( std::get<idx>( x ), std::get<idx>( y ) )... );
    }
    Kernels kernels;
};

/*-----------------------------------------------------------------------------
  convolutional kernel
-----------------------------------------------------------------------------*/
template<class Kernel> struct convolution {
    template<class R1, class R2>
    __GENERIC__ auto operator() ( R1 && range1, R2 && range2 ) {
        float sum = 0;
        for ( auto const & i : range1 ) {
            for ( auto const & j : range2 ) {
                sum += kernel( i, j );
            }
        }
        return sum;
    }
    Kernel kernel;
};

}

}

#endif
