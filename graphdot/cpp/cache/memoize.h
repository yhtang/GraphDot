#ifndef GRAPHDOT_CACHE_MEMOIZE_H_
#define GRAPHDOT_CACHE_MEMOIZE_H_

#include <iostream>
#include <csignal>
#include <memory>
#include <type_traits>
#include <cache/lru.h>
#include <unordered_map>
#include <misc/hash.h>

namespace graphdot { namespace cache {

template<class F> struct memoization_cache {

    F f;
    std::size_t max_size_;
    mutable std::shared_ptr<lru_cache_base> p_cache;

    memoization_cache( F && function, std::size_t max_size ) : f( std::forward<F>( function ) ), max_size_( max_size ) {}

    template<class ...Args> decltype( auto ) operator() ( Args && ... args ) const {

        using key_type   = std::tuple<std::remove_cv_t<std::remove_reference_t<Args> >...>; // deliberately turn all refs to copies
        using value_type = std::invoke_result_t < F, Args && ... >;      // preserve value category
        using cache_type = lru_cache<key_type, value_type>;

        if ( p_cache == nullptr ) p_cache = std::make_shared<cache_type>( max_size_ );
        cache_type * p = dynamic_cast<cache_type *>( p_cache.get() );
        if ( p == nullptr ) {
            std::cerr << "Mismatching arguments to a memoized function!\n" << std::flush;
            raise( SIGSEGV );
        }
        cache_type & cache = *p;

        auto key = key_type( args... );
        if ( !cache.has( key ) ) cache.insert( key, f( std::forward<Args>( args )... ) );
        return cache.get_copy( key );
    }
};

template<class F> inline auto memoize( F && f, std::size_t max_size = 0 ) {
    return memoization_cache<F>( std::forward<F>( f ), max_size );
}

} }

#endif
