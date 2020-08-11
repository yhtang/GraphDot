#ifndef GRAPHDOT_CUDA_UTIL_DEVICE_H_
#define GRAPHDOT_CUDA_UTIL_DEVICE_H_

namespace graphdot {

namespace cuda {

constexpr static int warp_size = 32;

__inline__ __device__ int laneid() {
    #if 0
    int laneid;
    asm volatile( "mov.s32 %0, %%laneid;" : "=r"( laneid ) );
    return laneid;
    #else
    return threadIdx.x % 32;
    #endif
}

__inline__ __device__ unsigned int lanemask_lt() {
    unsigned int lanemask32_lt;
    asm volatile("mov.u32 %0, %%lanemask_lt;" : "=r"(lanemask32_lt));
    return (lanemask32_lt);
}

template<class T> __device__ T warp_sum( T value ) {
#pragma unroll
    for ( int p = ( warp_size >> 1 ); p >= 1 ; p >>= 1 ) value += __shfl_xor_sync( 0xFFFFFFFF, value, p );
    return value;
}

template<class T> __device__ T block_sum( T value ) {
    __shared__ T bsum;
    if ( threadIdx.x == 0 ) bsum = 0;

    #pragma unroll
    for ( int p = ( warp_size >> 1 ); p >= 1 ; p >>= 1 ) value += __shfl_xor_sync( 0xFFFFFFFF, value, p );

    __syncthreads();
    if ( laneid() == 0 ) atomicAdd( &bsum, value );
    __syncthreads();
    return bsum;
}

template<class T> __inline__ __device__ T block_vdotv( T const * __restrict x, T const * __restrict y, int const N ) {
    T wsum = 0;
    for ( int i = threadIdx.x; i < N; i += blockDim.x ) wsum += x[i] * y[i];
    return block_sum( wsum );
}

#define ATOMIC_OP(name, op) \
__inline__ __device__ float atomic##name(float * ptr, float const value) {\
    auto ptr_as_int = reinterpret_cast<int *>(ptr);\
    auto old = *ptr_as_int;\
    auto assumed = old;\
    \
    do {\
        assumed = old;\
        old = atomicCAS(\
            ptr_as_int,\
            assumed,\
            __float_as_int(op(value, __int_as_float(assumed)))\
        );\
    } while (assumed != old);\
    \
    return __int_as_float(old);\
}

ATOMIC_OP(Min, min) // provides atomicMin
ATOMIC_OP(Max, max) // provides atomicMax

}

}

#endif
