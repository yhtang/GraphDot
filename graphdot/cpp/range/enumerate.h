#ifndef GRAPHDOT_RANGE_ENUMERATE_H_
#define GRAPHDOT_RANGE_ENUMERATE_H_

#include <tuple>
#include <utility>

namespace graphdot {

namespace internal {

template<class R>
struct enumerate_range {

	enumerate_range( R && r ) : range( std::forward<R>(r) ) {}

	template<class It>
    struct enumerate_iterator {
        It it;
        std::size_t i = 0;
        enumerate_iterator( It itr ) : it( itr ) {}
        auto operator * () {
            return std::forward_as_tuple( i, *it );
        }
        auto operator ++ () {
            ++it;
            ++i;
            return *this;
        }
        auto operator != ( enumerate_iterator const & other ) const {
            return it != other.it;
        }
    };

    // TODO: implement const-correctness
    auto begin() { return enumerate_iterator( std::begin( range ) ); }
    auto end  () { return enumerate_iterator( std::end  ( range ) ); }

    R range;
};

}

template<class R> auto enumerate( R && range ) {
    return internal::enumerate_range<R>( std::forward<R>( range ) );
}

}

#endif
