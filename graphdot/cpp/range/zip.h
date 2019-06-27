#ifndef GRAPHDOT_MISC_ZIP_H_
#define GRAPHDOT_MISC_ZIP_H_

#include <tuple>
#include <utility>

namespace graphdot {

template<class ...T> constexpr inline void expand( T && ... ignore ) noexcept {}

namespace internal {

template<class TR, class I, I ... idx>
struct zipped_range {

    zipped_range( TR && tr ) : tuple_of_ranges( std::forward<TR>( tr ) ) {}

    template<class ItrTuple>
    struct zipped_iterator {
        ItrTuple it;
        zipped_iterator( ItrTuple && ituple ) : it( std::forward<ItrTuple>( ituple ) ) {}
        auto operator * () {
            return std::make_tuple( *( std::get<idx>( it ) )... );
        }
        auto operator ++ () {
            expand( ++std::get<idx>( it )... );
            return *this;
        }
        auto operator != ( zipped_iterator const & other ) const { // here != means that all components has to be !=, for meaningful test of != end()
            return ( true && ... && ( std::get<idx>( it ) != std::get<idx>( other.it ) ) );
        }
    };

    // TODO: implement const-correctness
    auto begin() { return zipped_iterator( std::make_tuple( std::begin( std::get<idx>( tuple_of_ranges ) )... ) ); }
    auto end  () { return zipped_iterator( std::make_tuple( std::end  ( std::get<idx>( tuple_of_ranges ) )... ) ); }

    TR tuple_of_ranges;
};

template<class T, class I, I ... idx> auto make_zipped_range( T && tuple_of_ranges, std::integer_sequence<I, idx...> ) {
    return zipped_range<T, std::size_t, idx...>( std::forward<T>( tuple_of_ranges ) );
}

}

template<class ...R> auto zip( R && ... ranges ) {
    return internal::make_zipped_range( std::make_tuple( std::forward<R>( ranges )... ), std::make_index_sequence<sizeof...( R )>() );
}

}

#endif
