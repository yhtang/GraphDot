#ifndef GRAPHDOT_ARRAY_H_
#define GRAPHDOT_ARRAY_H_

namespace graphdot {

template<class type, int count>
struct array {
    using element_type = type;
    constexpr static int size = count;

    element_type _data[size];

    __host__ __device__ __inline__
    array() = default;

    template<class T>
    __host__ __device__ __inline__
    array(T const value) {
        #pragma unroll (size)
        for(int i = 0; i < size; ++i) {
            _data[i] = value;
        }
    }

    __host__ __device__ __inline__
    array & operator = (array const & other) {
        #pragma unroll (size)
        for(int i = 0; i < size; ++i) {
            _data[i] = other._data[i];
        }
        return *this;
    }

    __host__ __device__ __inline__
    element_type & operator [] (int i) {return _data[i];}
};

// Gracefully handles 0-length arrays
template<class type>
struct array<type, 0> {
    using element_type = type;
    constexpr static int size = 0;

    __host__ __device__ __inline__
    array() = default;

    template<class T>
    __host__ __device__ __inline__
    array(T const value) {}

    __host__ __device__ __inline__
    array & operator = (array const & other) {
        return *this;
    }

    __host__ __device__ __inline__
    auto operator [] (int i) {return element_type {};}
};

}

#endif
