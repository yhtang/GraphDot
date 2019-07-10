#ifndef GRAPHDOT_RANGE_INDEX_H_
#define GRAPHDOT_RANGE_INDEX_H_

namespace graphdot {

namespace internal {

template<class T>
struct index_range {

    index_range( T n ) : size( n ) {}

    struct index_iterator {
        T i;
        index_iterator( T _1 ) : i(_1) {}
        auto operator * () const {
            return i;
        }
        auto operator ++ () {
            ++i;
            return *this;
        }
        auto operator != ( index_iterator const & other ) const {
            return i != other.i;
        }
    };

    auto begin() const { return index_iterator( 0    ); }
    auto end  () const { return index_iterator( size ); }

    T size;
};

}

template<class T> auto range( T n ) {
	assert( n >= 0 );
    return internal::index_range<T>( n );
}

}

#endif
