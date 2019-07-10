#ifndef GRAPHDOT_CUDA_BALLOC_H_
#define GRAPHDOT_CUDA_BALLOC_H_

#include <iostream>
#include <vector>
#include <cuda_runtime.h>

namespace graphdot {

namespace cuda {

struct belt_allocator {

    std::size_t slab_size;

    enum class AllocMode { Device = 0, Managed = 1, Pinned = 2 } alloc_mode;

    belt_allocator( std::size_t slab_size = 1024*1024, AllocMode alloc_mode = AllocMode::Managed ) :
        slab_size( slab_size ),
        alloc_mode( alloc_mode )
    {
        current = allocate_slab();
    }

    ~belt_allocator() {
        cudaDeviceSynchronize();
        for( auto p: used ) cudaFree( p );
    }

    constexpr static std::size_t pad( std::size_t n, std::size_t align ) { return ( ( n + align - 1 ) / align ) * align; }

    void * operator() ( std::size_t size, std::size_t alignment = 32 ) {
        p_head = pad( p_head, alignment );
        if ( p_head + size > slab_size ) {
            used.push_back( current );
            current = allocate_slab();
            p_head = 0;
        }
        char * allocation = current + p_head;
        p_head += size;
        return allocation;
    }

    std::vector<char *> used;
    char * current = nullptr;
    std::size_t p_head = 0;

protected:
    char * allocate_slab() {
        char * p;
        if ( alloc_mode == AllocMode::Device ) cudaMalloc( &p, slab_size );
        else if ( alloc_mode == AllocMode::Managed ) cudaMallocManaged( &p, slab_size );
        else if ( alloc_mode == AllocMode::Pinned  ) cudaMallocHost( &p, slab_size );
        if ( cudaPeekAtLastError() ) {
            std::cout << "Error: " << cudaGetErrorString( cudaPeekAtLastError() ) << std::endl;
            throw std::bad_alloc();
        }
        return p;
    }
};

}

}

#endif
