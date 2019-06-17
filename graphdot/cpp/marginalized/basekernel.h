#ifndef GRAPHDOT_MARGINALIZED_BASEKERNEL_H
#define GRAPHDOT_MARGINALIZED_BASEKERNEL_H

#include <utility>
#include <string>
#include <cmath>
#include <math/utility.h>

namespace graphdot {

namespace kernel {

/*-----------------------------------------------------------------------------
  constant kernel
-----------------------------------------------------------------------------*/
struct constant {
    template<class X, class Y>
    __host__ __device__ auto operator() ( X const & x, Y const & y ) const {
        return constant;
    }
    float constant;
};

/*-----------------------------------------------------------------------------
  Kronecker delta/identity kernel
-----------------------------------------------------------------------------*/
struct kronecker_delta {
    template<class X, class Y>
    __host__ __device__ auto operator() ( X const & x, Y const & y ) const {
        return x == y ? hi : lo;
    }
    float lo, hi;
};

/*-----------------------------------------------------------------------------
  Gaussian kernel
-----------------------------------------------------------------------------*/
struct square_exponential {
    template<class X, class Y>
    __host__ __device__ auto operator() ( X const & x, Y const & y ) const {
        return expf( ( x - y ) * ( x - y ) * hl2inv );
    }
    float length_scale;
};

/*-----------------------------------------------------------------------------
  tensor product kernel
-----------------------------------------------------------------------------*/
template<class ...> struct tensor_product;

template<class Kernels, class I, I ...idx> struct tensor_product<Kernels, std::integer_sequence<I, idx...> > {
    template<class ...Args> tensor_product( Args && ... args ) : kernels( std::forward<Args>( args )... ) {}
    #ifdef __cpp_fold_expressions
    template<class X, class Y> double operator() ( X && x, Y && y ) const {
        return ( 1.0 * ... * std::get<idx>( kernels )( std::get<idx>( x ), std::get<idx>( y ) ) );
    }
    #endif
    Kernels kernels;
};

template<class ...K>
auto make_tensor_product_kernel( K && ... kernels ) {
    return tensor_product<std::tuple<K...>, std::make_index_sequence<sizeof...( K )> >( std::forward<K>( kernels )... );
}

/*-----------------------------------------------------------------------------
  convolutional kernel
-----------------------------------------------------------------------------*/
template<class Kernel> struct convolutional_kernel {
    convolutional_kernel( Kernel && kernel ) : kernel( std::forward<Kernel>( kernel ) ) {}
    template<class R1, class R2> double operator() ( R1 && range1, R2 && range2 ) {
        double sum = 0;
        for ( auto const & i : range1 ) {
            for ( auto const & j : range2 ) {
                sum += kernel( i, j );
            }
        }
        return sum;
    }
    Kernel kernel;
};

template<class K>
auto make_convolutional_kernel( K && kernel ) {
    return convolutional_kernel<K>( std::forward<K>( kernel ) );
}

}

}

#endif
