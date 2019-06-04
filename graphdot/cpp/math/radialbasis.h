#ifndef GRAPHDOT_CPP_MATH_RADIAL_BASIS_H
#define GRAPHDOT_CPP_MATH_RADIAL_BASIS_H

#include <cmath>
#include <cassert>
#include <limits>
#include <math/utility.h>

namespace graphdot { namespace rbf {

inline double gaussian( double r, double l, unsigned int derive = 0, unsigned int normalize_for = 0 ) {
    /*
    $\exp( -1/2 r^2 / l^2 )$, a.k.a. the Gaussian kernel

    Parameters
    ----------
    r: scalar or array_like
        radial distance
    l: scalar or array_like
        length scale
    derive: 0 or 1 or 2, optional
        computes the n-th order derivative
    normalize_for: None or 1 or 2 or 3, optional
        whether to normalize the integral of the function integral from 0 to \infty

    Returns
    ----------
    same shape as r
        basis function value at each element of r
    */

    assert( l > 0 );

    r = std::abs( r );

    double f;
    if ( derive == 0 ) {
        f = std::exp( -0.5 * ipow( r, 2 ) / ipow( l, 2 ) );
    } else if ( derive == 1 ) {
        f = -r * std::exp( -0.5 * ipow( r, 2 ) / ipow( l, 2 ) ) / ipow( l, 2 );
    } else if ( derive == 2 ) {
        f = -std::exp( -0.5 * ipow( r, 2 ) / ipow( l, 2 ) ) / ipow( l, 2 ) + std::exp( -0.5 * ipow( r, 2 ) / ipow( l, 2 ) ) * ipow( r, 2 ) / ipow( l, 4 );
    } else {
        return std::numeric_limits<double>::quiet_NaN();
    }

    if ( normalize_for == 0 ) {
        return f;
    } else if ( normalize_for == 3 ) {
        return f / std::sqrt( ipow( 2.0 * M_PI, 3 ) ) / ipow( l, 3 );
    } else if ( normalize_for == 2 ) {
        return f / ( 2.0 * M_PI ) / ipow( l, 2 );
    } else if ( normalize_for == 1 ) {
        return f / std::sqrt( 0.5 * M_PI ) / l;
    } else {
    	return std::numeric_limits<double>::quiet_NaN();
    }
}

inline double tent( double r, double s, double cutoff, unsigned int derive = 0, unsigned int normalize_for = 0 ) {
    /*
    $( 1 - r / cutoff )^s$, a.k.a. the generalized 1-over-r kernel

    Parameters
    ----------
    r: scalar or array_like
        radial distance
    s: scalar or array_like
        power of exponentiation
    cutoff: scalar or array_like
        cutoff distance beyond which the function decays to 0
    derive: 0 or 1 or 2, optional
        computes the n-th order derivative
    normalize_for: None or 1 or 2 or 3, optional
        whether to normalize the integral of the function integral from 0 to \infty

    Returns
    ----------
    same shape as r
        basis function value at each element of r
    */

    r = std::abs( r );
    auto h = std::max( 1 - r / cutoff, 0.0 );

    double f;
    if ( derive == 0 ) {
    	assert( s > 0 );
    	f = ipow( h, s );
    } else if ( derive == 1 ) {
    	assert( s > 1 );
        f = -s * ipow( h, s-1 ) / cutoff;
    } else if ( derive == 2 ) {
    	assert( s > 2 );
        f = s * ( s - 1 ) * ipow( h, s-2 ) / ipow( cutoff, 2 );
    } else {
    	return std::numeric_limits<double>::quiet_NaN();
    }

    if ( normalize_for == 0 ) {
        return f;
    } else if ( normalize_for == 3 ) {
        return f / ( 8.0 * M_PI * ipow( cutoff, 3 ) / ( ipow( s, 3 ) + 6.0*ipow( s, 2 ) + 11.0*s + 6.0) );
    } else if ( normalize_for == 2 ) {
        return f / ( 2.0 * M_PI * ipow( cutoff, 2 ) / ( ipow( s, 2 ) + 3.0*s + 2.0 ) );
    } else if ( normalize_for == 1 ) {
        return f / ( 2.0 * cutoff / ( s + 1.0 ) );
    } else {
    	return std::numeric_limits<double>::quiet_NaN();
    }
}
        //def Laplace( r, l, derive = 0, normalize_for = None ):
//    """
//    $\exp( -r / l )$, a.k.a. the Laplace kernel
//
//    Parameters
//    ----------
//    r: scalar or array_like
//        radial distance
//    l: scalar or array_like
//        length scale
//    derive: 0 or 1 or 2, optional
//        computes the n-th order derivative
//    normalize_for: None or 1 or 2 or 3, optional
//        whether to normalize the integral of the function integral from 0 to \infty
//
//    Returns
//    ----------
//    same shape as r
//        basis function value at each element of r
//    """
//
//    assert( l > 0 )
//
//    r = numpy.abs( r )
//
//    if derive == 0:
//        f = std::exp( -r / l )
//    elif derive == 1:
//        f = -std::exp( -r / l ) / l
//    elif derive == 2:
//        f = std::exp( -r / l ) / ipow( l, 2 )
//    else:
//        raise ValueError( 'Invalid argument: derive' )
//
//    if normalize_for is None:
//        return f
//    elif normalize_for == 3:
//        return f / ( 8.0 * M_PI * ipow( l, 3 ) )
//    elif normalize_for == 2:
//        return f / ( 2.0 * M_PI * ipow( l, 2 ) )
//    elif normalize_for == 1:
//        return f / ( 2.0 * l )
//    else:
//        raise ValueError( 'Invalid argument: normalize_for' )import numpy
//
//def Polynomial( r, a, b, cutoff, derive = 0, normalize_for = None ):
//    """
//    $ -b * ( 1 - r / rc )^a + a * ( 1 - r / rc )^b $, a two-term polynomial kernel with compact support
//
//    Parameters
//    ----------
//    r: scalar or array_like
//        radial distance
//    a: scalar, > b
//        power of first term, must be larger than the second term b
//    b: scalar, >= 2
//        power of second term, must be no less than 2 to ensure kernel continuity
//    cutoff: scalar or array_like
//        cutoff distance beyond which the function decays to 0
//    derive: 0 or 1 or 2, optional
//        computes the n-th order derivative
//    normalize_for: None or 1 or 2 or 3, optional
//        whether to normalize the integral of the function integral from 0 to \infty
//
//    Returns
//    ----------
//    same shape as r
//        basis function value at each element of r
//    """
//
//    assert( a >  b )
//    assert( b >= 2 )
//
//    r = numpy.abs( r )
//    h = numpy.maximum( 1 - r / cutoff, 0.0 )
//
//    if derive == 0:
//        f = -b * h**a + a * h**b
//    elif derive == 1:
//        f = b * a * h**(a-1) - a * b * h**(b-1)
//    elif derive == 2:
//        f = -b * a * (a-1) * h**(a-2) + a * b * (b-1) * h**(b-2)
//    else:
//        raise ValueError( 'Invalid argument: derive' )
//
//    if normalize_for is None:
//        return f
//    elif normalize_for == 3:
//        return f / ( 8.0 * M_PI * ipow( cutoff, 3 ) * ( a / (ipow( b, 3 ) + 6.0*ipow( b, 2 ) + 11.0*b + 6.0) - b / (ipow( a, 3 ) + 6.0*ipow( a, 2 ) + 11.0*a + 6.0) ) )
//    elif normalize_for == 2:
//        return f / ( 2.0 * M_PI * ipow( cutoff, 2 ) * ( a / (ipow( b, 2 ) + 3.0*b + 2.0) - b / (ipow( a, 2 ) + 3.0*a + 2.0) ) )
//    elif normalize_for == 1:
//        return f / ( 2.0 * cutoff * ( a / (b + 1.0) - b / (a + 1.0) ) )
//    else:
//        raise ValueError( 'Invalid argument: normalize_for' )import numpy


} }

#endif
