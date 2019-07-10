#ifndef GRAPHDOT_MARGINALIZED_KRONECKER_SQUARE_EXPONENTIAL_H_
#define GRAPHDOT_MARGINALIZED_KRONECKER_SQUARE_EXPONENTIAL_H_

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cuda/balloc.h>
#include <cuda/util_host.h>
#include <cuda/util_device.h>
#include <kernel/common.h>
#include <misc/hash.h>
#include <misc/format.h>

namespace graphdot {

namespace kernel {

namespace detail {

/*-----------------------------------------------------------------------------
Block CG solver
-----------------------------------------------------------------------------*/
// solve the MLGK linear system using matrix-free CG
// where the matvec operation is done in 8x8 tiles
template<class Graph>
struct octile_block_solver {

    /*  Each simple octile consists of 64 elemented arranged in column-major storage
        =========================================
        |  0 |  8 | 16 | 24 | 32 | 40 | 48 | 56 |
        -----------------------------------------
        |  1 |  9 | 17 | 25 | 33 | 41 | 49 | 57 |
        -----------------------------------------
        |  2 | 10 | 18 | 26 | 34 | 42 | 50 | 58 |
        -----------------------------------------
        |  3 | 11 | 19 | 27 | 35 | 43 | 51 | 59 |
        -----------------------------------------
        |  4 | 12 | 20 | 28 | 36 | 44 | 52 | 60 |
        -----------------------------------------
        |  5 | 13 | 21 | 29 | 37 | 45 | 53 | 61 |
        -----------------------------------------
        |  6 | 14 | 22 | 30 | 38 | 46 | 54 | 62 |
        -----------------------------------------
        |  7 | 15 | 23 | 31 | 39 | 47 | 55 | 63 |
        =========================================
        The right-hand side is a 64-element vector
     */

    using edge_t = typename Graph::edge_t;

    constexpr static int octile_w = 8;
    constexpr static int octile_h = 8;

    struct octile { // maps a piece of shared memory as an octile for matvec computation

        edge_t * const _data;

        constexpr static int size_bytes = octile_w * octile_h * sizeof( edge_t );

        constexpr __inline__ __device__ __host__ octile( void * ptr ) : _data( reinterpret_cast<edge_t *>( ptr ) ) {}

        constexpr __inline__ __device__ __host__ edge_t & operator() ( int i, int j ) {
            return _data[ i + j * octile_h ];
        }

        constexpr __inline__ __device__ __host__ edge_t & operator() ( int i ) {
            return _data[ i ];
        }
    };

    struct rhs { // maps a piece of shared memory as 4 hexdectors for matvec computation

        float * const _data;

        constexpr static int size_bytes = octile_w * octile_w * sizeof( float );

        constexpr __inline__ __device__ rhs( void * ptr ) : _data( reinterpret_cast<float *>( ptr ) ) {}

        constexpr __inline__ __device__ float & operator() ( int j1, int j2 ) {
            return _data[ j1 * octile_w + j2 ];
        }
    };

    constexpr static int shmem_bytes_per_warp = octile::size_bytes * 2 + rhs::size_bytes;

    // compute submatrix inner prpduct
    template<class EdgeKernel>
    __inline__ __device__ static void mmv_octile (
        EdgeKernel ke,
        const int i1_upper,
        const int i1_lower,
        const int i2,
        octile octile1,
        octile octile2,
        rhs rhs,
        int j1_margin,
        float & sum_upper,
        float & sum_lower
    ) {
        #pragma unroll (octile_w)
        for ( int j1 = 0; j1 < octile_w && j1 < j1_margin; ++j1 ) {
            auto e1_upper = octile1( i1_upper, j1 );
            auto e1_lower = octile1( i1_lower, j1 );
            #pragma unroll (octile_w)
            for ( int j2 = 0; j2 < octile_w; ++j2 ) {
                auto e2 = octile2( i2, j2 );
                auto r  = rhs( j1, j2 );
                sum_upper -= ke( e1_upper, e2 ) * r;
                sum_lower -= ke( e1_lower, e2 ) * r;
            }
        }
    }

    template<class VertexKernel, class EdgeKernel>
    __inline__ __device__ static auto compute(
        VertexKernel   kv,
        EdgeKernel     ke,
        Graph const    g1,
        Graph const    g2,
        block_scratch  scratch,
        char * const   p_shared,
        const float    s,
        const float    q ) {

        using namespace graphdot::cuda;

        const int warp_id_local = threadIdx.x / warp_size;
        const int warp_num_local = blockDim.x / warp_size;
        const int lane = laneid();
        const int n1 = g1.padded_size();
        const int n2 = g2.padded_size();
        const int N  = n1 * n2;

        for ( int i = threadIdx.x; i < N; i += blockDim.x ) {
            int i1 = i / n2;
            int i2 = i % n2;
            float d1 = g1.degree[ i1 ];
            float d2 = g2.degree[ i2 ];
            scratch.x( i ) = 0;
            float r = d1 * d2 * q * q;
            scratch.r( i ) = r;
            scratch.p( i ) = r;
            scratch.Ap( i ) = d1 * d2 / kv( g1.vertex[ i1 ], g2.vertex[ i2 ] ) * r;
        }
        __syncthreads();

        auto rTr = block_vdotv( scratch.r(), scratch.r(), N );

        int k;
        for ( k = 0; k < N; ++k ) {

            #if 0
            __syncthreads();
            if ( threadIdx.x == 0 && blockIdx.x == 0 ) {
                for ( int ij = 0; ij < N; ++ij ) {
                    printf( "iteration %d solution x[%d] = %.7f\n", k, ij, scratch.x( ij ) );
                }
            }
            #endif

            const int i1_upper =   lane               / octile_h;
            const int i1_lower = ( lane + warp_size ) / octile_h;
            const int i2       =   lane               % octile_h;

            //if ( lane == 0 ) printf("%s %d\n", __FILE__, __LINE__ );

            // Ap = A * p, off-diagonal part
            for( int O1 = 0; O1 < g1.n_octile; O1 += warp_num_local ) {

                const int nt1 = min( g1.n_octile - O1, warp_num_local );

                if ( warp_id_local < nt1 ) {
                    // load the first submatrix into shared memory, stored in col-major layout
                    //if ( lane == 0 ) printf("loading left octile %d\n", O1 + warp_id_local );
                    auto o1 = g1.octile[ O1 + warp_id_local ];
                    octile octile1 { p_shared + warp_id_local * shmem_bytes_per_warp };
                    octile1( lane             ) = o1.elements[ lane             ];
                    octile1( lane + warp_size ) = o1.elements[ lane + warp_size ];
                }

                __syncthreads();

                for(int O2 = 0; O2 < g2.n_octile; O2 += warp_num_local ) {

                    const int nt2 = min( g2.n_octile - O2, warp_num_local );

                    if ( warp_id_local < nt2 ) {
                        //if ( lane == 0 ) printf("loading right octile %d\n", O2 + warp_id_local );
                        auto o2 = g2.octile[ O2 + warp_id_local ];
                        octile octile2 { p_shared + warp_id_local * shmem_bytes_per_warp + octile::size_bytes };
                        // load the second submatrix into cache, stored in col-major layout
                        octile2( lane             ) = o2.elements[ lane             ];
                        octile2( lane + warp_size ) = o2.elements[ lane + warp_size ];
                    }

                    __syncthreads();

                    for( int t = warp_id_local; t < nt1 * nt2; t += warp_num_local ) {

                        const int p1 = t / nt2;
                        const int p2 = t % nt2;

                        //if ( lane == 0 ) printf("computing %d-%d\n", p1, p2 );

                        auto o1 = g1.octile[ O1 + p1 ];
                        const int I1 = o1.upper;
                        const int J1 = o1.left;
                        auto o2 = g2.octile[ O2 + p2 ];
                        const int I2 = o2.upper;
                        const int J2 = o2.left;

                        octile octile1 { p_shared + p1 * shmem_bytes_per_warp };
                        octile octile2 { p_shared + p2 * shmem_bytes_per_warp + octile::size_bytes };
                        rhs    rhs     { p_shared + warp_id_local * shmem_bytes_per_warp + octile::size_bytes + octile::size_bytes };

                        // load RHS
                        int j1 = lane / octile_w;
                        int j2 = lane % octile_w;
                        rhs( j1,                        j2 ) = scratch.p( ( J1 + j1                        ) * n2 + ( J2 + j2 ) );
                        rhs( j1 + warp_size / octile_w, j2 ) = scratch.p( ( J1 + j1 + warp_size / octile_w ) * n2 + ( J2 + j2 ) );

                        float sum_upper = 0, sum_lower = 0;
                        mmv_octile( ke, i1_upper, i1_lower, i2, octile1, octile2, rhs, g1.n_vertex - J1, sum_upper, sum_lower );


                        atomicAdd( &scratch.Ap( ( I1 + i1_upper ) * n2 + ( I2 + i2 ) ), sum_upper );
                        atomicAdd( &scratch.Ap( ( I1 + i1_lower ) * n2 + ( I2 + i2 ) ), sum_lower );
                    }

                    __syncthreads();
                }
            }

            __syncthreads();

            // alpha = rTr / dot( p, Ap );
            auto pAp = block_vdotv( scratch.p(), scratch.Ap(), N );
            auto alpha = rTr / pAp;

            // x = x + alpha * p;
            // r = r - alpha * Ap;
            for ( int i = threadIdx.x; i < N; i += blockDim.x ) {
                scratch.x( i ) += alpha * scratch.p ( i );
                scratch.r( i ) -= alpha * scratch.Ap( i );
            }
            //__syncthreads(); // not needed

            auto rTr_next = block_vdotv( scratch.r(), scratch.r(), N );

            if ( rTr_next < float( 1e-16 ) * N * N ) break;

            auto beta = rTr_next / rTr;

            // p = r + beta * p;
            for ( int i = threadIdx.x; i < N; i += blockDim.x ) {
                //scratch.p(i) = scratch.r(i) + beta * scratch.p(i);
                int i1 = i / n2;
                int i2 = i % n2;
                float p = scratch.r( i ) + beta * scratch.p( i );
                scratch.p( i ) = p;
                scratch.Ap( i ) = g1.degree[ i1 ] * g2.degree[ i2 ] / kv( g1.vertex[ i1 ], g2.vertex[ i2 ] ) * p;
            }
            __syncthreads();

            rTr = rTr_next;
        }

        float R = 0;
        for ( int i = threadIdx.x; i < N; i += blockDim.x ) {
            R += s * s * scratch.x( i );
        }
        R = warp_sum( R );
        __shared__ float block_R;
        if ( threadIdx.x == 0 ) block_R = 0;
        __syncthreads();
        if ( laneid() == 0 ) atomicAdd( &block_R, R );
        __syncthreads();
        #if 0
        __syncthreads();
        if ( threadIdx.x == 0 ) {
            printf( "Converged after %d iterations\n", k );
            printf( "R(sum) = %.7f\n", block_R );
            #if 0
            for ( int ij = 0; ij < N; ++ij ) {
                printf( "solution x[%d] = %.7f\n", ij, scratch.x( ij ) );
            }
            #endif
        }
        __syncthreads();
        #endif
        dotjob retval;
        retval.out.r = block_R;
        retval.out.it = k;
        return retval;
    }
};

}

template<class Graph, class VertexKernel, class EdgeKernel>
__global__ void mlgk_batch_conjugate_gradient(
    VertexKernel      kv,
    EdgeKernel        ke,
    Graph const      * graphs,
    block_scratch    * scratch,
    dotjob           * jobs,
    unsigned int     * i_job_global,
    const unsigned     n_jobs,
    const float        s,
    const float        q
) {

    using kernel_t = detail::octile_block_solver<Graph>;

    extern __shared__ char shmem[];

    __shared__ unsigned int i_job;

    while( true ) {
        if ( threadIdx.x == 0 ) i_job = atomicInc( i_job_global, 0xFFFFFFFF );
        __syncthreads();

        if ( i_job >= n_jobs ) break;

        auto out = kernel_t::compute( kv, ke, graphs[ jobs[i_job].in.i ], graphs[ jobs[i_job].in.j ], scratch[ blockIdx.x ], shmem, s, q );
        if ( threadIdx.x == 0 ) jobs[i_job] = out;
        __syncthreads();
    }
}

struct marginalized_kronecker_sqexp {

    using real_t      = float;
    using scratch_t   = block_scratch;

    struct kernel_vertex {
        real_t lo;
        __inline__ __host__ __device__ kernel_vertex( real_t v_lo ) : lo( v_lo ) {}
        __inline__ __host__ __device__ kernel_vertex( kernel_vertex const & ) = default;
        template<class X> __inline__ __device__ real_t operator() ( X a, X b ) const {
            return ( a == b ? 1 : lo );
        }
    };

    struct kernel_edge {
        real_t hl2inv;
        __inline__ __host__ __device__ kernel_edge( real_t length_scale ) : hl2inv( -0.5f / ( length_scale * length_scale ) ) {}
        __inline__ __host__ __device__ kernel_edge( kernel_edge const & ) = default;
        __inline__ __device__ real_t operator() ( float2 a, float2 b ) const {
            return a.x * b.x * __expf( ( a.y - b.y ) * ( a.y - b.y ) * hl2inv );
        }
    };

    struct graph_t {

        using deg_t  = real_t;
        using vert_t = int;
        using edge_t = float2;

        struct octile_t {
            int upper, left;
            edge_t * elements;
        };

        constexpr static float eps = 1e-14;

        int n_vertex, n_octile;
        deg_t    * degree;
        vert_t   * vertex;
        octile_t * octile;

        constexpr __inline__ __host__ __device__ int padded_size() const {
            return ( n_vertex + 7 ) & ~7U;
        }

        graph_t( pybind11::tuple pygraph, cuda::belt_allocator & alloc, pybind11::dict hyperparameters ) {

            pybind11::list vertices = pygraph[0];
            pybind11::list edges    = pygraph[1];

            n_vertex = vertices.size();

            degree = ( deg_t  * ) alloc( padded_size() * sizeof( deg_t  ) );
            vertex = ( vert_t * ) alloc( padded_size() * sizeof( vert_t ) );

            cudaMemset( degree, 0, padded_size() * sizeof( deg_t  ) );
            cudaMemset( vertex, 0, padded_size() * sizeof( vert_t ) );

            std::unordered_map<std::pair<int, int>, octile_t> octile_table;

            for ( auto & item: edges ) {
                auto e = pybind11::cast<pybind11::tuple>( item );
                auto i = pybind11::cast<int>  ( e[0] );
                auto j = pybind11::cast<int>  ( e[1] );
                auto w = pybind11::cast<float>( e[2] );
                auto l = pybind11::cast<float>( e[3] );
                degree[ i ] += w;
                degree[ j ] += w;

                if ( w > eps ) {
                    int r = i % 8;
                    int c = j % 8;
                    int I = i - r;
                    int J = j - c;

                    if ( octile_table.find( std::make_pair( I, J ) ) == octile_table.end() ) {
                        octile_t tile1;
                        tile1.upper = I;
                        tile1.left  = J;
                        tile1.elements = ( edge_t * ) alloc( sizeof( edge_t ) * 8 * 8 );
                        cudaMemset( tile1.elements, 0, sizeof( edge_t ) * 8 * 8 );
                        octile_table.emplace( std::make_pair( I, J ), tile1 );

                        if ( I != J ) {
                            octile_t tile2;
                            tile2.upper = J;
                            tile2.left  = I;
                            tile2.elements = ( edge_t * ) alloc( sizeof( edge_t ) * 8 * 8 );
                            cudaMemset( tile2.elements, 0, sizeof( edge_t ) * 8 * 8 );
                            octile_table.emplace( std::make_pair( J, I ), tile2 );
                        }
                    }

                    octile_table.find( std::make_pair( I, J ) )->second.elements[ r + c * 8 ] = make_float2( w, l );
                    octile_table.find( std::make_pair( J, I ) )->second.elements[ c + r * 8 ] = make_float2( w, l );
                }
            }

            auto q = pybind11::cast<float>( hyperparameters["stopping_probability"] );

            for ( auto & item: vertices ) {
                auto v = pybind11::cast<pybind11::tuple>( item );
                auto i = pybind11::cast<int>( v[0] );
                auto l = pybind11::cast<int>( v[1] );
                vertex[i]  = l;
                degree[i] /= 1 - q;
            }

            n_octile = octile_table.size();
            octile = ( octile_t * ) alloc( sizeof( octile_t ) * n_octile );

            for ( int I = 0, k = 0; I < n_vertex; I += 8 ) {
                for ( int J = 0; J < n_vertex; J += 8 ) {
                    if ( octile_table.find( std::make_pair( I, J ) ) != octile_table.end() ) {
                        octile[ k++ ] = octile_table[ std::make_pair( I, J ) ];
                    }
                }
            }
        }
    };

    int device;
    int block_per_sm;
    int block_size;

    cudaDeviceProp gpu_properties;

    template<class T>
    struct device_ptr {
        using element_type = T;
        constexpr static std::size_t element_size = sizeof(T);
        T * ptr = nullptr;
        std::size_t size = 0, capacity_ = 0;
        ~device_ptr() { if ( ptr ) cudaFree( ptr ); }
        void resize( std::size_t new_size ) {
            if ( new_size > capacity_ ) {
                if ( ptr ) cudaFree( ptr );
                while( new_size > capacity_ ) capacity_ = capacity_ * 2 + 1;
                cudaMalloc( &ptr, capacity_ * sizeof(T) );
            }
            size = new_size;
        }
        inline operator T*() const {
            return ptr;
        }
    };

    cuda::belt_allocator allocator;

    std::size_t scratch_size = 0;
    device_ptr<scratch_t>    dev_scratch;
    device_ptr<dotjob>       dev_jobs;
    device_ptr<unsigned int> dev_i_job_global;
    device_ptr<graph_t>      dev_graph_list;

    marginalized_kronecker_sqexp( pybind11::dict runtime_config ) {
        // setup CUDA context
        device       = pybind11::cast<int>( runtime_config["device"      ] );
        block_per_sm = pybind11::cast<int>( runtime_config["block_per_sm"] );
        block_size   = pybind11::cast<int>( runtime_config["block_size"  ] );

        cuda::detect_cuda( device );
        cuda::verify( cudaGetDeviceProperties( &gpu_properties, device ) );
        cuda::verify( cudaSetDevice( device ) );
    }

    void allocate_scratch( std::size_t n, std::size_t max_size ) {
        cudaDeviceSynchronize();

        allocator.~belt_allocator();

        std::size_t scratch_allocation_granularity = 128 * 1024 * 1024; // 128 MB

        new( &allocator ) cuda::belt_allocator( scratch_allocation_granularity );

        std::vector<scratch_t> scratches;
        for ( std::size_t i = 0; i < n; ++i ) scratches.emplace_back( max_size, allocator );

        dev_scratch.resize( n );

        cudaMemcpy( dev_scratch, scratches.data(), dev_scratch.size * dev_scratch.element_size, cudaMemcpyDefault );

        cudaDeviceSynchronize();
    }

    auto compute( pybind11::dict hyperparameters, pybind11::list jobs, pybind11::list graph_list ) {

        kernel_vertex kv( pybind11::cast<float>( hyperparameters["vertex_baseline_similarity" ] ) );
        kernel_edge   ke( pybind11::cast<float>( hyperparameters["edge_length_scale"          ] ) );
        float s = pybind11::cast<float>( hyperparameters["starting_probability"       ] );
        float q = pybind11::cast<float>( hyperparameters["stopping_probability"       ] );

        cuda::sync_and_peek( __FILE__, __LINE__ );

        std::size_t job_count = jobs.size();
        dev_jobs.resize( job_count );

        cuda::sync_and_peek( __FILE__, __LINE__ );

        dev_i_job_global.resize( 1 );

        cuda::sync_and_peek( __FILE__, __LINE__ );

        cudaMemset( dev_i_job_global, 0, dev_i_job_global.size * dev_i_job_global.element_size );

        cuda::sync_and_peek( __FILE__, __LINE__ );

        std::vector<dotjob> job_list_cpu;
        for(auto const &item: jobs) {
            auto pair = pybind11::cast<pybind11::tuple>(item);
            dotjob job;
            job.in.i = pybind11::cast<int>( pair[0] );
            job.in.j = pybind11::cast<int>( pair[1] );
            job_list_cpu.push_back( job );
        }
        cudaMemcpy( dev_jobs, job_list_cpu.data(), dev_jobs.size * dev_jobs.element_size, cudaMemcpyDefault );

        cuda::sync_and_peek( __FILE__, __LINE__ );

        dev_graph_list.resize( graph_list.size() );
        std::vector<graph_t> graph_list_cpu;
        int max_graph_size = 0;
        for(auto const &pygraph: graph_list ) {
            graph_list_cpu.push_back( pybind11::cast<graph_t>( pygraph ) );
            max_graph_size = std::max<int>( max_graph_size, graph_list_cpu.back().padded_size() );
        }
        cudaMemcpy( dev_graph_list, graph_list_cpu.data(), dev_graph_list.size * dev_graph_list.element_size, cudaMemcpyDefault );

        cuda::sync_and_peek( __FILE__, __LINE__ );

        std::size_t launch_block_count = gpu_properties.multiProcessorCount * block_per_sm;
        std::size_t shmem_bytes_per_block = detail::octile_block_solver<graph_t>::shmem_bytes_per_warp * block_size / cuda::warp_size;

        cuda::sync_and_peek( __FILE__, __LINE__ );

        if ( max_graph_size * max_graph_size > scratch_size ) {
            scratch_size = max_graph_size * max_graph_size;
            allocate_scratch( launch_block_count, scratch_size );
        }

        cuda::sync_and_peek( __FILE__, __LINE__ );

        std::cout << format("Derived launch parameters:" ) << std::endl;
        std::cout << format("------------------------------------------------------------------" ) << std::endl;
        std::cout << format("%-32s : %ld", "Blocks launched", launch_block_count ) << std::endl;
        std::cout << format("%-32s : %ld", "Shared memory per block", shmem_bytes_per_block ) << std::endl;
        std::cout << format("------------------------------------------------------------------" ) << std::endl;

        cuda::sync_and_peek( __FILE__, __LINE__ );

        mlgk_batch_conjugate_gradient<graph_t, kernel_vertex, kernel_edge>
        <<< launch_block_count, block_size, shmem_bytes_per_block >>>
        ( kv,
          ke,
          dev_graph_list,
          dev_scratch,
          dev_jobs,
          dev_i_job_global,
          dev_jobs.size,
          s,
          q );

        cuda::sync_and_peek( __FILE__, __LINE__ );

        cudaMemcpyAsync( job_list_cpu.data(), dev_jobs, dev_jobs.size * dev_jobs.element_size, cudaMemcpyDefault );

        cuda::verify( ( cudaDeviceSynchronize() ) );

        std::vector<float> result;
        for(auto const &j: job_list_cpu) result.push_back( j.out.r );

        return result;
    }
};

}

}

#endif
