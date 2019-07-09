#ifndef GRAPHDOT_MATH_LINALG_H_
#define GRAPHDOT_MATH_LINALG_H_

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <iostream>
#include <type_traits>

namespace graphdot { namespace linalg {

template<typename _Scalar, int _Options, typename _StorageIndex>
inline auto rowsum( Eigen::SparseMatrix<_Scalar, _Options, _StorageIndex> const & M ) {
    Eigen::Matrix<_Scalar, Eigen::Dynamic, 1> s;
    s = s.Zero( M.rows() );
    for ( std::size_t k = 0; k < M.outerSize(); ++k )
        for ( typename Eigen::SparseMatrix<_Scalar, _Options, _StorageIndex>::InnerIterator it( M, k ); it; ++it )
            s( it.row() ) += it.value();
    return s;
}

template<class Derived>
inline auto rowsum( Eigen::MatrixBase<Derived> const & M ) {
    return M.rowwise().sum().eval();
}

template<class Derived>
inline auto diag( Eigen::MatrixBase<Derived> const & d, std::bool_constant<true> sparse ) {
    using scalar = typename Eigen::MatrixBase<Derived>::Scalar;
    std::vector<Eigen::Triplet<scalar> > triplet;
    for ( std::size_t i = 0; i < d.size(); ++i ) triplet.emplace_back( i, i, d( i ) );
    Eigen::SparseMatrix<scalar> D( d.size(), d.size() );
    D.setFromTriplets( triplet.begin(), triplet.end() );
    return D;
}

template<typename _Scalar, int _Rows, int _Options, int _MaxRows, int _MaxCols>
inline auto diag( Eigen::Matrix<_Scalar, _Rows, 1, _Options, _MaxRows, _MaxCols> const & d, std::bool_constant<false> sparse ) {
    return Eigen::Matrix<_Scalar, _Rows, _Rows, _Options, _MaxRows, _MaxRows>( d.asDiagonal() );
}

template<class Matrix>
inline auto pinvh( Matrix const & M, double rcond = 1e-15 ) {

	Eigen::SelfAdjointEigenSolver<Matrix> solver( M );
    auto w = solver.eigenvalues();
    auto V = solver.eigenvectors();

    auto threshold = w.maxCoeff() * rcond;
    std::size_t n = ( w.array() > threshold ).count() ;
    decltype(w) wm( n );
    decltype(V) Vm( M.rows(), n );
    std::size_t p = 0;
    for ( std::size_t i = 0; i < w.size(); ++i ) {
        if ( w( i ) > threshold ) {
            wm( p ) = w( i );
            Vm.col( p ) = V.col( i );
            ++p;
        }
    }

    auto logdet = wm.array().log().sum();
    decltype(V) Minv = Vm * wm.cwiseInverse().asDiagonal() * Vm.transpose();

    return std::make_pair( std::move( Minv ), logdet );
}

} }

#endif
