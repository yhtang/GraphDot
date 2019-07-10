#ifndef GRAPHDOT_MISC_HASH_H_
#define GRAPHDOT_MISC_HASH_H_

#include <functional>
#include <Eigen/Core>

namespace graphdot {

constexpr inline std::size_t hash_combine( std::size_t h ) noexcept { return h; }

template<class ...T>
constexpr inline std::size_t hash_combine( std::size_t h1, std::size_t h2, T ... ts ) noexcept {
	return hash_combine( h1 ^ ( h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2) ), ts...  );
}

template<class T> constexpr inline std::size_t get_hash( T const & t ) {
	return std::hash<T>{}(t);
}

}

namespace std {

template<class ...T> struct hash<std::tuple<T...> > {

	template<class Tuple, class I, I ... Idx>
	inline std::size_t __hash( Tuple const &tuple, std::integer_sequence<I, Idx...> ) const noexcept {
		return graphdot::hash_combine( graphdot::get_hash( std::get<Idx>(tuple) )... );
	}

	inline std::size_t operator() ( std::tuple<T...> const & tpl ) const noexcept {
		return __hash( tpl, std::make_index_sequence<sizeof...(T)>() );
	}
};

template<class A, class B> struct hash<pair<A,B> > {
	inline std::size_t operator() ( pair<A, B> const & p ) const noexcept {
		return graphdot::hash_combine( graphdot::get_hash( p.first ), graphdot::get_hash( p.second ) );
	}
};

template<class T> struct hash<vector<T> > {
	inline std::size_t operator() ( vector<T> const & items ) const noexcept {
		std::size_t seed = 0;
		for(auto const &t: items ) seed = graphdot::hash_combine( seed, graphdot::get_hash( t ) );
		return seed;
	}
};

template<typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
struct hash<Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols> > {
	inline std::size_t operator() ( Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols> const & m ) const noexcept {
		std::size_t seed = 0;
		for( std::size_t j = 0; j < m.cols(); j++) {
			for(std::size_t i = 0; i < m.rows(); ++i) {
				seed = graphdot::hash_combine( seed, graphdot::get_hash( m(i,j) ) );
			}
		}
		return seed;
	}
};

}

#endif
