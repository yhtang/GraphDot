#ifndef GRAPHDOT_CACHE_LRU_H_
#define GRAPHDOT_CACHE_LRU_H_

#include <unordered_map>
#include <list>

namespace graphdot {

namespace cache {

struct lru_cache_base { virtual ~lru_cache_base() {} }; // added to facilitate implementation of memoize()

template<class Key, class Value> struct lru_cache : lru_cache_base {

    using storage_type = std::list<std::pair<Key, Value> >;
    using lut_type = std::unordered_map<Key, typename storage_type::iterator>;

    mutable storage_type storage;
    lut_type     lookuptable;
    std::size_t  max_size_;

    lru_cache( std::size_t max_size  = 0 ) : max_size_( max_size ) {}

    // returns existence boolean, does not increase entry lifetime
    bool has( Key const & key ) {
        return lookuptable.find( key ) != lookuptable.end();
    }

    void insert( Key const & key, Value && value ) {
        auto itr = lookuptable.find( key );
        if ( itr != lookuptable.end() ) {
            itr->second->second = std::forward<Value>( value );
            storage.splice( storage.begin(), storage, itr->second );
        } else {
            storage.emplace_front( std::make_pair( key, std::forward<Value>( value ) ) );
            lookuptable[key] = storage.begin();
            prune();
        }
    }

    void insert( Key const & key, Value const & value ) {
        auto itr = lookuptable.find( key );
        if ( itr != lookuptable.end() ) {
            itr->second->second = value;
            storage.splice( storage.begin(), storage, itr->second );
        } else {
            storage.emplace_front( std::make_pair( key, value ) );
            lookuptable[key] = storage.begin();
            prune();
        }
    }

    Value get_copy( Key const & key ) const {
        return get_ref( key );
    }

    Value & get_ref( Key const & key ) {
        auto itr = lookuptable.find( key );
        storage.splice( storage.begin(), storage, itr->second );
        return itr->second->second;
    }

    Value const & get_ref( Key const & key ) const {
        auto itr = lookuptable.find( key );
        storage.splice( storage.begin(), storage, itr->second );
        return itr->second->second;
    }

    std::size_t prune() {
        if ( max_size_ == 0 || storage.size() < max_size_ ) return 0;

        std::size_t n_removed = 0;
        while ( storage.size() > max_size_ ) {
            lookuptable.erase( storage.back().first );
            storage.pop_back();
            ++n_removed;
        }
        return n_removed;
    }

    std::size_t set_max_size( std::size_t max_size ) {
        max_size_ = max_size;
        return prune();
    }
};

}

}

#endif
