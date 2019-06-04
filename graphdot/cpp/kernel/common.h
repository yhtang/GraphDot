#ifndef GRAPHDOT_CPP_KERNEL_COMMON_H_
#define GRAPHDOT_CPP_KERNEL_COMMON_H_

#include <cuda/balloc.h>

namespace graphdot {

namespace kernel {

struct dotjob {
    union {
        struct { int i, j; } in;
        struct { float r; int it; } out;
    };
};

struct block_scratch {

    float * ptr;
    int stride;

    block_scratch( int max_size, cuda::belt_allocator & alloc ) {
        stride = ( ( max_size + 15 ) / 16 ) * 16;
        ptr = ( float * ) alloc( stride * sizeof( float ) * 4 );
    }

    __host__ __device__ __inline__ constexpr float * x  () { return ptr + stride * 0; }
    __host__ __device__ __inline__ constexpr float * r  () { return ptr + stride * 1; }
    __host__ __device__ __inline__ constexpr float * p  () { return ptr + stride * 2; }
    __host__ __device__ __inline__ constexpr float * Ap () { return ptr + stride * 3; }
    __host__ __device__ __inline__ constexpr float & x  ( int i ) { return x()[i]; }
    __host__ __device__ __inline__ constexpr float & r  ( int i ) { return r()[i]; }
    __host__ __device__ __inline__ constexpr float & p  ( int i ) { return p()[i]; }
    __host__ __device__ __inline__ constexpr float & Ap ( int i ) { return Ap()[i]; }
};

}

}

#endif
