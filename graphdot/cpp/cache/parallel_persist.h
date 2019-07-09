#ifndef GRAPHDOT_CACHE_PARALLEL_PERSIST_H_
#define GRAPHDOT_CACHE_PARALLEL_PERSIST_H_

#include <atomic>
#include <mutex>
#include <iostream>
#include <fstream>
#include <string>
#include <unordered_map>
#include <type_traits>
#include <thread>
#include <ext/json/json.hpp>
#include <misc/timer.h>

namespace graphdot { namespace cache {

template<class F, class KeyMaker> struct parallel_persist_cache {

    std::atomic<bool> write_protection = false;

    parallel_persist_cache( F && f, std::string filename, KeyMaker && keymaker ) :
        f_( std::forward<F>( f ) ),
        filename_( filename ),
        keymaker_( std::forward<KeyMaker>( keymaker ) )
    {
        std::ifstream in( filename_ );
        if ( !in.eof() && !in.fail() ) in >> data;
    }
    ~parallel_persist_cache() {
        dump();
    }

    template<class ...Args> std::invoke_result_t<F, Args...> inline operator() ( Args && ... args ) const {
        std::string key = keymaker_( args... );

        {
            std::lock_guard<std::mutex> guard( mutex );
            auto itr = data.find( key );
            if ( itr != data.end() ) return itr.value();
        }

        ++dirty;
        auto val = f_( std::forward<Args>( args )... );

        std::lock_guard<std::mutex> guard( mutex );
        auto [ itr, ignore ] = data.emplace( key, val );
        return val;
    }

    bool dump() {
        if ( dirty && !write_protection ) {
            std::ofstream( filename_ ) << data.dump( 1 );
            dirty = 0;
            return true;
        } else {
            return false;
        }
    }

protected:
    F f_;
    std::string filename_;
    KeyMaker keymaker_;
    mutable json::json data;
    mutable std::atomic<std::size_t> dirty = 0;
    mutable std::mutex mutex;
};

struct concatenation_key_maker {

    template<class T, class ...Ts> inline std::string operator() ( T const & t, Ts const & ... ts ) const {
        return ( std::to_string( t ) + ... + ( sep + std::to_string( ts ) ) );
    }

    std::string sep = "$";
};

template<class F, class KeyMaker = concatenation_key_maker>
inline auto parallel_persist( F && f, std::string filename, KeyMaker && keymaker = KeyMaker {} ) {
    return parallel_persist_cache<F, KeyMaker>( std::forward<F>( f ), filename, std::forward<KeyMaker>( keymaker ) );
}

} }

#endif
