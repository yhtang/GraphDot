#ifndef GRAPHDOT_MARGINALIZED_KERNEL_H_
#define GRAPHDOT_MARGINALIZED_KERNEL_H_

#include <cuda/balloc.h>
#include <cuda/util_host.h>
#include <cuda/util_device.h>
#include <misc/format.h>

namespace graphdot {

namespace marginalized {

struct job_t {
    int i, j;
    float *vr;
};

struct block_scratch {

    float * ptr;
    int stride;

    block_scratch (int max_size, cuda::belt_allocator & alloc) {
        stride = ((max_size + 15) / 16) * 16;
        ptr = (float *) alloc (stride * sizeof (float) * 4);
    }

    __host__ __device__ __inline__ constexpr float * x  () { return ptr + stride * 0; }
    __host__ __device__ __inline__ constexpr float * r  () { return ptr + stride * 1; }
    __host__ __device__ __inline__ constexpr float * p  () { return ptr + stride * 2; }
    __host__ __device__ __inline__ constexpr float * Ap () { return ptr + stride * 3; }
    __host__ __device__ __inline__ constexpr float & x  (int i) { return x()[i]; }
    __host__ __device__ __inline__ constexpr float & r  (int i) { return r()[i]; }
    __host__ __device__ __inline__ constexpr float & p  (int i) { return p()[i]; }
    __host__ __device__ __inline__ constexpr float & Ap (int i) { return Ap()[i]; }
};

template<class Node, class Edge> struct graph_t {

    using deg_t  = float;
    using node_t = Node;
    using edge_t = Edge;
    using octile_t = struct {
        int upper, left;
        union {
            std::uint64_t nzmask; // row-major! in constrat to the column-major layout of elements
            unsigned char nzmask_bytes[8];
        };
        edge_t * elements;
    };

    constexpr static float eps = 1e-14;

    int n_node, n_octile;
    deg_t   *  degree;
    node_t  *  vertex;
    octile_t * octile;

    constexpr __inline__ __host__ __device__ int padded_size() const {
        return (n_node + 7) & ~7U;
    }
};

/*-----------------------------------------------------------------------------
Block CG solver: solve the MLGK linear system using matrix-free CG where
the matvec operations are done in 8x8 tiles.
-----------------------------------------------------------------------------*/
template<class Graph>
struct octile_block_solver {
    /*  Each simple octile consists of 64 elemented put in column-major format
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
        The right-hand side is a 64-element vector */

    using edge_t = typename Graph::edge_t;

    constexpr static int octile_w = 8;
    constexpr static int octile_h = 8;

    struct octile { // maps a piece of shared memory as an octile for matvec computation

        edge_t * const _data;

        constexpr static int size_bytes = octile_w * octile_h * sizeof (edge_t);

        constexpr __inline__ __device__ __host__ octile (void * ptr) : _data (reinterpret_cast<edge_t *> (ptr)) {}

        constexpr __inline__ __device__ __host__ edge_t & operator() (int i, int j) {
            return _data[ i + j * octile_h ];
        }

        constexpr __inline__ __device__ __host__ edge_t & operator() (int i) {
            return _data[i];
        }
    };

    struct rhs { // maps a piece of shared memory as 4 hexdectors for matvec computation

        float * const _data;

        constexpr static int size_bytes = octile_w * octile_w * sizeof (float);

        constexpr __inline__ __device__ rhs (void * ptr) : _data (reinterpret_cast<float *> (ptr)) {}

        constexpr __inline__ __device__ float & operator() (int j1, int j2) {
            return _data[ j1 * octile_w + j2 ];
        }
    };

    constexpr static int shmem_bytes_per_warp = octile::size_bytes * 2 + rhs::size_bytes;

    // compute submatrix inner prpduct
    template<class EdgeKernel>
    __inline__ __device__ static void mmv_octile (
        const int i1_upper,
        const int i1_lower,
        const int i2,
        uint nzmask1_uplo,
        const uint nzmask2,
        octile octile1,
        octile octile2,
        rhs rhs,
        int j1_margin,
        float & sum_upper,
        float & sum_lower
    ) {
        #pragma unroll (octile_w)
        for (int j1 = 0; j1 < octile_w && j1 < j1_margin; ++j1) {
            auto e1_upper = octile1 (i1_upper, j1);
            auto e1_lower = octile1 (i1_lower, j1);
            bool m1_upper = nzmask1_uplo & 0x1;
            bool m1_lower = nzmask1_uplo & 0x100;

            #if 1
            #define ITERATION(j2, mask) \
                auto e2_##j2 = octile2 (i2, j2);\
                auto r_##j2  = rhs (j1, j2);\
                bool m2_##j2 = nzmask2 & mask;\
                if (m1_upper && m2_##j2) sum_upper -= EdgeKernel::compute(e1_upper, e2_##j2) * r_##j2;\
                if (m1_lower && m2_##j2) sum_lower -= EdgeKernel::compute(e1_lower, e2_##j2) * r_##j2;

            ITERATION(0, 0x1)
            ITERATION(1, 0x2)
            ITERATION(2, 0x4)
            ITERATION(3, 0x8)
            ITERATION(4, 0x10)
            ITERATION(5, 0x20)
            ITERATION(6, 0x40)
            ITERATION(7, 0x80)
            #undef ITERATION
            #else
            #pragma unroll (octile_w)
            for (int j2 = 0, mask = 1; j2 < octile_w; ++j2, mask <<= 1) {
                auto e2 = octile2 (i2, j2);
                auto r  = rhs (j1, j2);
                bool m2 = nzmask2 & mask;
                if (m1_upper && m2) {
                    sum_upper -= EdgeKernel::compute(e1_upper, e2) * r;
                }
                if (m1_lower && m2) {
                    sum_lower -= EdgeKernel::compute(e1_lower, e2) * r ;
                }
            }
            #endif

            nzmask1_uplo >>= 1;
        }
    }

    template<class NodeKernel, class EdgeKernel>
    __inline__ __device__ static void compute(
        Graph const    g1,
        Graph const    g2,
        block_scratch  scratch,
        char * const   p_shared,
        const float    q,
        const float    q0,
        const int      lmin,
        float *        vr) {

        using namespace graphdot::cuda;

        const int warp_id_local = threadIdx.x / warp_size;
        const int warp_num_local = blockDim.x / warp_size;
        const int lane = laneid();
        const int n1 = g1.padded_size();
        const int n2 = g2.padded_size();
        const int N  = n1 * n2;

        for (int i = threadIdx.x; i < N; i += blockDim.x) {
            int i1 = i / n2;
            int i2 = i % n2;
            float d1 = g1.degree[i1] / (1 - q);
            float d2 = g2.degree[i2] / (1 - q);
            scratch.x (i) = 0;
            float r = d1 * d2 * q * q / (q0 * q0);
            // printf("r[%d] = %f, D[%d] = %f, deg[%d] = %f, deg[%d] = %f\n", i, r, i, d1 * d2, i1, g1.degree[i1], i2, g2.degree[i2]);
            scratch.r (i) = r;
            scratch.p (i) = r;
            scratch.Ap (i) = (i1 < g1.n_node && i2 < g2.n_node) ? d1 * d2 / NodeKernel::compute(g1.vertex[i1], g2.vertex[i2]) * r : 0.f;
            // if (i1 < g1.n_node && i2 < g2.n_node) printf("Kv([%ld,%ld], [%ld,%ld]]) = %f\n", g1.vertex[i1].hybridization, g1.vertex[i1].charge, g2.vertex[i2].hybridization, g2.vertex[i2].charge, NodeKernel::compute(g1.vertex[i1], g2.vertex[i2]));
        }
        __syncthreads();

        auto rTr = block_vdotv (scratch.r(), scratch.r(), N);

        int k;
        for (k = 0; k < N; ++k) {

            #if 0
            __syncthreads();
            if (threadIdx.x == 0) {
                for (int ij = 0; ij < N; ++ij) {
                    int i1 = ij / n2;
                    int i2 = ij % n2;
                    if (i1 < g1.n_node && i2 < g2.n_node)
                        printf ("iteration %d begin solution x[%d, %d] = %.7f, Ap[%d, %d] = %.7f\n", k, i1, i2, scratch.x(ij), i1, i2, scratch.Ap(ij));
                }
            }
            #endif

            const int i1_upper =  lane              / octile_h;
            const int i1_lower = (lane + warp_size) / octile_h;
            const int i2       =  lane              % octile_h;

            // Ap = A * p, off-diagonal part
            for (int O1 = 0; O1 < g1.n_octile; O1 += warp_num_local) {

                const int nt1 = min (g1.n_octile - O1, warp_num_local);

                if (warp_id_local < nt1) {
                    // load the first submatrix into shared memory, stored in col-major layout
                    //if ( lane == 0 ) printf("loading left octile %d\n", O1 + warp_id_local );
                    auto o1 = g1.octile[ O1 + warp_id_local ];
                    octile octile1 { p_shared + warp_id_local * shmem_bytes_per_warp };
                    octile1 (lane            ) = o1.elements[lane            ];
                    octile1 (lane + warp_size) = o1.elements[lane + warp_size];
                }

                __syncthreads();

                for (int O2 = 0; O2 < g2.n_octile; O2 += warp_num_local) {

                    const int nt2 = min (g2.n_octile - O2, warp_num_local);

                    if (warp_id_local < nt2) {
                        //if ( lane == 0 ) printf("loading right octile %d\n", O2 + warp_id_local );
                        auto o2 = g2.octile[ O2 + warp_id_local ];
                        octile octile2 {p_shared + warp_id_local * shmem_bytes_per_warp + octile::size_bytes};
                        // load the second submatrix into cache, stored in col-major layout
                        octile2 (lane            ) = o2.elements[lane            ];
                        octile2 (lane + warp_size) = o2.elements[lane + warp_size];
                    }

                    __syncthreads();

                    for (int t = warp_id_local; t < nt1 * nt2; t += warp_num_local) {

                        const int p1 = t / nt2;
                        const int p2 = t % nt2;

                        //if ( lane == 0 ) printf("computing %d-%d\n", p1, p2 );

                        auto o1 = g1.octile[ O1 + p1 ];
                        const int I1 = o1.upper;
                        const int J1 = o1.left;
                        // const std::uint64_t nzmask1 = o1.nzmask;
                        auto o2 = g2.octile[ O2 + p2 ];
                        const int I2 = o2.upper;
                        const int J2 = o2.left;
                        // const std::uint64_t nzmask2 = o2.nzmask;

                        octile octile1 {p_shared + p1 * shmem_bytes_per_warp};
                        octile octile2 {p_shared + p2 * shmem_bytes_per_warp + octile::size_bytes};
                        rhs    rhs     {p_shared + warp_id_local * shmem_bytes_per_warp + octile::size_bytes + octile::size_bytes};

                        // load RHS
                        int j1 = lane / octile_w;
                        int j2 = lane % octile_w;
                        rhs (j1,                        j2) = scratch.p ((J1 + j1                       ) * n2 + (J2 + j2));
                        rhs (j1 + warp_size / octile_w, j2) = scratch.p ((J1 + j1 + warp_size / octile_w) * n2 + (J2 + j2));

                        float sum_upper = 0, sum_lower = 0;
                        mmv_octile<EdgeKernel>(i1_upper, i1_lower, i2, o1.nzmask_bytes[i1_upper] | ( o1.nzmask_bytes[i1_lower] << 8 ), o2.nzmask_bytes[i2], octile1, octile2, rhs, g1.n_node - J1, sum_upper, sum_lower);
                        // printf("threadIdx %d sum_upper  %d %f sum_lower %d %f\n", threadIdx.x, (I1 + i1_upper) * n2 + (I2 + i2), sum_upper, (I1 + i1_lower) * n2 + (I2 + i2), sum_lower);


                        atomicAdd(&scratch.Ap((I1 + i1_upper) * n2 + (I2 + i2)), sum_upper);
                        atomicAdd(&scratch.Ap((I1 + i1_lower) * n2 + (I2 + i2)), sum_lower);
                    }

                    __syncthreads();
                }
            }

            __syncthreads();

            // alpha = rTr / dot( p, Ap );
            auto pAp = block_vdotv (scratch.p(), scratch.Ap(), N);
            auto alpha = rTr / pAp;

            // x = x + alpha * p;
            // r = r - alpha * Ap;
            for (int i = threadIdx.x; i < N; i += blockDim.x) {
                scratch.x (i) += alpha * scratch.p (i);
                scratch.r (i) -= alpha * scratch.Ap (i);
            }
            //__syncthreads(); // not needed

            auto rTr_next = block_vdotv (scratch.r(), scratch.r(), N);

            if (rTr_next < float (1e-20) * N * N) break;

            auto beta = rTr_next / rTr;

            // p = r + beta * p;
            for (int i = threadIdx.x; i < N; i += blockDim.x) {
                // scratch.p(i) = scratch.r(i) + beta * scratch.p(i);
                int i1 = i / n2;
                int i2 = i % n2;
                float p = scratch.r (i) + beta * scratch.p (i);
                scratch.p(i) = p;
                auto d1 = g1.degree[i1] / (1 - q);
                auto d2 = g2.degree[i2] / (1 - q);
                scratch.Ap(i) = (i1 < g1.n_node && i2 < g2.n_node) ? d1 * d2 / NodeKernel::compute(g1.vertex[i1], g2.vertex[i2]) * p : 0.f;
            }
            __syncthreads();

            rTr = rTr_next;
        }
        __syncthreads();

        for (int i = threadIdx.x; i < N; i += blockDim.x) {
            int i1 = i / n2;
            int i2 = i % n2;
            if (i1 < g1.n_node && i2 < g2.n_node) {
                auto r = scratch.x(i);
                if (lmin == 1) {
                    r -=  NodeKernel::compute(g1.vertex[i1], g2.vertex[i2]) * q * q / (q0 * q0);
                }
                vr[i1 * g2.n_node + i2] = r;
            }
        }

        #if 0
        __syncthreads();
        float R = 0;
        for (int i = threadIdx.x; i < N; i += blockDim.x) {
            R += scratch.x (i);
        }
        R = warp_sum (R);
        __shared__ float block_R;
        if (threadIdx.x == 0) block_R = 0;
        __syncthreads();
        if (laneid() == 0) atomicAdd (&block_R, R);
        __syncthreads();
        if (threadIdx.x == 0) {
            printf ("sum(R) = %.7f\n", block_R);
            printf ("Converged after %d iterations\n", k);
            for (int ij = 0; ij < N; ++ij) {
                printf ("solution x[%d] = %.7f\n", ij, scratch.x (ij));
            }
        }
        __syncthreads();
        #endif
    }
};

}

}

#endif
