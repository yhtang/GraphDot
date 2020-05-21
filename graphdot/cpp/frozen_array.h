#ifndef GRAPHDOT_CONV_H_
#define GRAPHDOT_CONV_H_

#include <type_traits>
#include "numpy_type.h"

namespace graphdot {

namespace numpy_type {

template<class T>
struct frozen_array {
    using element_type = T;
    using reference_type = std::add_lvalue_reference_t<element_type>;
    using const_reference_type = std::add_lvalue_reference_t<std::add_const_t<element_type>>;
    using pointer_type = std::add_pointer_t<element_type>;
    using size_type = numpy_type::int32;
    pointer_type * _data = nullptr;
    size_type size = 0;

    struct const_iterator {
        pointer_type * _ptr;

        void operator ++ () {++_ptr;}

        const_reference_type operator * () {
            return *_ptr;
        }

        bool operator != (const_iterator const &other) {
            return _ptr != other._ptr;
        }
    };

    const_iterator begin() const {
        return const_iterator {_ptr};
    }

    const_iterator end() const {
        return const_iterator {_ptr + size};
    }
};

}

}

#endif