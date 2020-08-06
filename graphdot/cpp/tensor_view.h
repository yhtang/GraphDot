#ifndef GRAPHDOT_TENSOR_VIEW_H_
#define GRAPHDOT_TENSOR_VIEW_H_

#include <type_traits>

namespace graphdot {

template<class type, int ndim>
struct tensor_view_t {
    using size_type = std::uint32_t;
    using element_type = std::remove_const_t<type>;
    using pointer_type = std::add_pointer_t<element_type>;

    pointer_type const _ptr;
    size_type const _shape[ndim];

    template<class ...Int>
    __host__ __device__ __inline__
    tensor_view_t(pointer_type ptr, Int ... shape):
        _ptr(ptr), _shape {shape...} {}

    template<class ...Int>
    __host__ __device__ __inline__
    pointer_type at(Int ... indices) {
        static_assert(sizeof...(indices) == ndim);
        // multi-index address is essentially polynomial evaluation
        size_type idx[ndim] {indices...};
        size_type offset = 0;
        for(int i = ndim - 1; i >= 0; --i) {
            offset = offset * _shape[i] + idx[i];
        }
        return _ptr + offset;
    }

    template<class ...Int>
    __host__ __device__ __inline__
    element_type & operator () (Int ... indices) {
        return *at(indices...);
    }
};

template<class T, class ... Int>
__host__ __device__ __inline__
auto tensor_view(T * ptr, Int ... shape) {
    return tensor_view_t<T, sizeof...(Int)>(ptr, shape...);
}

}

#endif
