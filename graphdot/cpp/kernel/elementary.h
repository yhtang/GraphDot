#ifndef GRAPHDOT_CPP_KERNEL_ELEMENTARY_H
#define GRAPHDOT_CPP_KERNEL_ELEMENTARY_H

#include <utility>
#include <string>
#include <cmath>
#include <math/utility.h>

namespace graphdot { namespace kernel {

/*-----------------------------------------------------------------------------
  tensor product kernel
-----------------------------------------------------------------------------*/
template<class ...> struct tensor_product;

template<class Kernels, class I, I ...idx> struct tensor_product<Kernels, std::integer_sequence<I, idx...> > {
	template<class ...Args> tensor_product( Args && ... args ) : kernels( std::forward<Args>(args)... ) {}
#ifdef __cpp_fold_expressions
	template<class X, class Y> double operator() ( X && x, Y && y ) const {
		return ( 1.0 * ... * std::get<idx>(kernels)( std::get<idx>(x), std::get<idx>(y) ) );
	}
#endif
	Kernels kernels;
};

template<class ...K>
auto make_tensor_product_kernel( K && ... kernels ) {
	return tensor_product<std::tuple<K...>, std::make_index_sequence<sizeof...(K)> >( std::forward<K>(kernels)... );
}

/*-----------------------------------------------------------------------------
  convolutional kernel
-----------------------------------------------------------------------------*/
template<class Kernel> struct convolutional_kernel {
    convolutional_kernel( Kernel && kernel ) : kernel( std::forward<Kernel>(kernel) ) {}
    template<class R1, class R2> double operator() ( R1 && range1, R2 && range2 ) {
        double sum = 0;
        for( auto const &i: range1 ) {
            for( auto const &j: range2 ) {
                sum += kernel( i, j );
            }
        }
        return sum;
    }
    Kernel kernel;
};

template<class K>
auto make_convolutional_kernel( K && kernel ) {
    return convolutional_kernel<K>( std::forward<K>(kernel) );
}

/*-----------------------------------------------------------------------------
  do-nothing kernel
-----------------------------------------------------------------------------*/
struct unity {
	template<class X, class Y> double operator() ( X const & x, Y const & y ) const {
		return 1.0;
	}
};

/*-----------------------------------------------------------------------------
  Kronecker delta/identity kernel
-----------------------------------------------------------------------------*/
struct kronecker_delta {
    kronecker_delta( double lo = 0 ) : lo(lo) {}

	template<class X, class Y> double operator() ( X const & x, Y const & y ) const {
		return x == y ? 1.0 : lo;
	}

	double lo = 0;
};

/*-----------------------------------------------------------------------------
  Gaussian kernel
-----------------------------------------------------------------------------*/
struct square_exponential {
	square_exponential( double length_scale ) : hl2inv( -0.5 / (length_scale*length_scale) ) {}

	template<class X, class Y> double operator() ( X const & x, Y const & y ) const {
		return std::exp( ( x - y ) * ( x - y ) * hl2inv );
	}

	double hl2inv;
};

/*-----------------------------------------------------------------------------
  outer product kernel
-----------------------------------------------------------------------------*/
/*
struct outer_product_kernel {
    template<class T1, class T2, int Rows1, int Rows2>
    Eigen::Matrix<std::common_type_t<T1,T2>,Rows1,Rows2> operator() (Eigen::Matrix<T1,Rows1,1> const & r1, Eigen::Matrix<T2,Rows2,1> const & r2) {
        return r1 * r2.transpose();
    }
};
*/
} }

#endif
