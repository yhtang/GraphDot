#ifndef GRAPHDOT_MATH_DISTANCE_H_
#define GRAPHDOT_MATH_DISTANCE_H_

#include <cmath>
#include <type_traits>
#include <Eigen/Dense>

namespace graphdot {

template<class ...> struct metric;

template<class T1, class T2> inline auto distance( T1 const & x, T2 const & y ) {
	return metric<std::common_type_t<T1, T2> >{}( x, y );
}

template<> struct metric<float>        { auto operator () ( float        x, float        y ) { return std::abs( x - y ); } };
template<> struct metric<double>       { auto operator () ( double       x, double       y ) { return std::abs( x - y ); } };
template<> struct metric<int>          { auto operator () ( int          x, int          y ) { return x > y ? x - y : y - x; } };
template<> struct metric<unsigned int> { auto operator () ( unsigned int x, unsigned int y ) { return x > y ? x - y : y - x; } };

template<typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
struct metric<Eigen::Matrix<_Scalar,_Rows,_Cols,_Options,_MaxRows,_MaxCols> > {
	using vect = Eigen::Matrix<_Scalar,_Rows,_Cols,_Options,_MaxRows,_MaxCols>;
	double operator () ( vect const & x, vect const & y ) {
		return ( x - y ).norm();
	}
};

}

#endif
