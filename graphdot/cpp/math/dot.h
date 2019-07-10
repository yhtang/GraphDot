#ifndef GRAPHDOT_MATH_DOT_H_
#define GRAPHDOT_MATH_DOT_H_

#include <cmath>
#include <type_traits>
#include <Eigen/Dense>

namespace graphdot {

template<class T1, class T2> inline auto dot( T1 const & x, T2 const & y ) {
    return x * y;
}

template<typename _Scalar1, int _Rows1, int _Cols1, int _Options1, int _MaxRows1, int _MaxCols1,
         typename _Scalar2, int _Rows2, int _Cols2, int _Options2, int _MaxRows2, int _MaxCols2>
inline auto dot( Eigen::Matrix<_Scalar1,_Rows1,_Cols1,_Options1,_MaxRows1,_MaxCols1> const &x,
          Eigen::Matrix<_Scalar1,_Rows1,_Cols1,_Options1,_MaxRows1,_MaxCols1> const &y ) {
    return x.dot( y );
};

}

#endif
