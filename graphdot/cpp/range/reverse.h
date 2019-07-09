#ifndef GRAPHDOT_RANGE_REVERSE_H_
#define GRAPHDOT_RANGE_REVERSE_H_

#include <iterator>

namespace graphdot {

namespace internal {

template<class R>
struct reverse_range {

	reverse_range( R && r ) : range( std::forward<R>(r) ) {}

    auto begin() { return std::rbegin( range ); }
    auto end  () { return std::rend  ( range ); }

    R && range;
};

}

template<class R> auto reverse( R && range ) {
    return internal::reverse_range<R>( std::forward<R>( range ) );
}

}

#endif
