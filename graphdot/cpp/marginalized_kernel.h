#ifndef GRAPHDOT_MARGINALIZED_KERNEL_H_
#define GRAPHDOT_MARGINALIZED_KERNEL_H_

#include "util_cuda.h"
#include "graph.h"

namespace graphdot {

namespace marginalized {

struct job_t {
    int     i, j;
    float * vr;
};

struct pcg_scratch_t {

    float * ptr;
    int     stride;

    pcg_scratch_t(pcg_scratch_t const & other) = default;

    __device__ __inline__ float * x() { return ptr + stride * 0; }
    __device__ __inline__ float * r() { return ptr + stride * 1; }
    __device__ __inline__ float * z() { return ptr + stride * 2; }
    __device__ __inline__ float * p() { return ptr + stride * 3; }
    __device__ __inline__ float * Ap() { return ptr + stride * 4; }
    __device__ __inline__ float & x(int i) { return x()[i]; }
    __device__ __inline__ float & r(int i) { return r()[i]; }
    __device__ __inline__ float & z(int i) { return z()[i]; }
    __device__ __inline__ float & p(int i) { return p()[i]; }
    __device__ __inline__ float & Ap(int i) { return Ap()[i]; }
};

/*-----------------------------------------------------------------------------
CG solver based on on-the-fly Kronecker product matrix-vector (XMV) operations
-----------------------------------------------------------------------------*/
template<class Graph> struct labeled_compact_block_dynsched_pcg {
    /*  Each octile contains up to 64 elemented in column-major format
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

    using graph_t   = Graph;
    using scratch_t = pcg_scratch_t;
    using node_t    = typename graph_t::node_t;
    using edge_t    = typename graph_t::edge_t;

    constexpr static int octile_w = 8;
    constexpr static int octile_h = 8;

    // maps a piece of shared memory as an octile for matvec computation
    struct octile {

        edge_t * const _data;

        constexpr static int size_bytes = octile_w * octile_h * sizeof(edge_t);

        __device__ __inline__ octile(void * ptr)
            : _data(reinterpret_cast<edge_t *>(ptr)) {}

        __device__ __inline__ edge_t & operator()(int i, int j) {
            return _data[i + j * octile_h];
        }

        __device__ __inline__ edge_t & operator()(int i) { return _data[i]; }
    };

    // maps a piece of shared memory as 4 hexdectors for matvec computation
    struct rhs {

        float * const _data;

        constexpr static int size_bytes = octile_w * octile_w * sizeof(float);

        __device__ __inline__ rhs(void * ptr)
            : _data(reinterpret_cast<float *>(ptr)) {}

        __device__ __inline__ float & operator()(int j1, int j2) {
            return _data[j1 * octile_w + j2];
        }
    };

    struct nzlist {
        using nzindex_t = int;

        nzindex_t * _data;

        constexpr static int size_bytes =
            octile_w * octile_h * sizeof(nzindex_t);

        __device__ __inline__ nzlist(void * ptr)
            : _data(reinterpret_cast<nzindex_t *>(ptr)) {}

        __device__ __inline__ nzindex_t & operator()(int i) { return _data[i]; }

        __device__ __inline__ nzindex_t const & operator()(int i) const {
            return _data[i];
        }
    };

    constexpr static int shmem_bytes_per_warp =
        3 * octile::size_bytes +
        1 * rhs::size_bytes + 
        2 * nzlist::size_bytes;

    template<class NodeKernel, class EdgeKernel>
    __inline__ __device__ static void compute(
        Graph const    g1,
        Graph const    g2,
        scratch_t      scratch,
        char * const   p_shared,
        const float    q,
        const float    q0,
        const int      lmin,
        float *        vr) {

        using namespace graphdot::cuda;

        const int warp_id_local  = threadIdx.x / warp_size;
        const int warp_num_local = blockDim.x / warp_size;
        const int lane           = laneid();
        const int n1             = g1.padded_size();
        const int n2             = g2.padded_size();
        const int N              = n1 * n2;

        octile octilex {p_shared + warp_id_local * shmem_bytes_per_warp};

        for (int i = threadIdx.x; i < N; i += blockDim.x) {
            int   i1 = i / n2;
            int   i2 = i % n2;
            float d1 = g1.degree[i1] / (1 - q);
            float d2 = g2.degree[i2] / (1 - q);
            // b  = Dx . qx
            float b = d1 * d2 * q * q / (q0 * q0);
            // r0 = b - A . x0
            //    = b
            float r0 = b;
            // z0 = M^-1 . r0
            //    = (Dx . Vx^-1)^-1 . r0
            //    = Vx . Dx^-1 . r0
            //    = Vx . Dx^-1 . b
            //    = Vx . Dx^-1 . Dx . qx
            //    = Vx . qx
            float z0 =
                i1 < g1.n_node && i2 < g2.n_node ?
                NodeKernel::compute( g1.node[i1], g2.node[i2] ) * q * q / (q0 * q0) :
                0;
            // x0 = 0
            scratch.x(i) = 0;
            scratch.r(i) = r0;
            scratch.z(i) = z0;
            // p0 = z0
            scratch.p(i) = z0;
            //Ap0 = diag(A . p0)
            //    = Dx . Vx^-1 . p0
            //    = Dx . Vx^-1 . Vx . qx
            //    = Dx . qx
            scratch.Ap(i) = d1 * d2 * q * q / (q0 * q0);
        }
        __syncthreads();

        auto rTz = block_vdotv(scratch.r(), scratch.z(), N);

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
            for (int O1 = 0; O1 < g1.n_tile; O1 += warp_num_local) {

                const int nt1 = min(g1.n_tile - O1, warp_num_local);

                if (warp_id_local < nt1) {
                    // load the first submatrix in compact format into shared memory
                    auto o1 = g1.octile[O1 + warp_id_local];
                    octile octile1 {p_shared + warp_id_local * shmem_bytes_per_warp + octilex.size_bytes};
                    nzlist nzlist1 {p_shared + warp_id_local * shmem_bytes_per_warp + octilex.size_bytes + octile1.size_bytes};

                    // expand into col-major dense ayout
                    const int nnz1 = __popcll(o1.nzmask);
                    if (lane             < nnz1) octilex(lane)             = o1.elements[lane];
                    if (lane + warp_size < nnz1) octilex(lane + warp_size) = o1.elements[lane + warp_size];

                    __syncwarp();

                    if (o1.nzmask_halves[0] & (1 << lane)) {
                        int src = __popc(o1.nzmask_halves[0] & lanemask_lt());
                        octile1(lane) = octilex(src);
                        nzlist1(src)  = lane;
                    }

                    if (o1.nzmask_halves[1] & (1 << lane)) {
                        int src = __popc(o1.nzmask_halves[1] & lanemask_lt()) +
                                  __popc(o1.nzmask_halves[0]);
                        octile1(lane + warp_size) = octilex(src);
                        nzlist1(src)              = lane + warp_size;
                    }
                }

                __syncthreads();

                for (int O2 = 0; O2 < g2.n_tile; O2 += warp_num_local) {

                    const int nt2 = min(g2.n_tile - O2, warp_num_local);

                    if (warp_id_local < nt2) {
                        // load the second submatrix in compact fornat into shared memory
                        auto o2 = g2.octile[O2 + warp_id_local];
                        octile octile2 {p_shared + warp_id_local * shmem_bytes_per_warp + octilex.size_bytes + octile::size_bytes + nzlist::size_bytes};
                        nzlist nzlist2 {p_shared + warp_id_local * shmem_bytes_per_warp + octilex.size_bytes + octile::size_bytes + nzlist::size_bytes + octile2.size_bytes};

                        // expand into col-major dense ayout
                        const int nnz2 = __popcll(o2.nzmask);
                        if (lane             < nnz2) octilex(lane)             = o2.elements[lane];
                        if (lane + warp_size < nnz2) octilex(lane + warp_size) = o2.elements[lane + warp_size];

                        __syncwarp();

                        if (o2.nzmask_halves[0] & (1 << lane)) {
                            int src = __popc(o2.nzmask_halves[0] & lanemask_lt());
                            octile2(lane) = octilex(src);
                            nzlist2(src)  = lane;
                        }

                        if (o2.nzmask_halves[1] & (1 << lane)) {
                            int src = __popc(o2.nzmask_halves[1] & lanemask_lt()) +
                                      __popc(o2.nzmask_halves[0]);
                            octile2(lane + warp_size) = octilex(src);
                            nzlist2(src)              = lane + warp_size;
                        }
                    }

                    __syncthreads();

                    for (int t = warp_id_local; t < nt1 * nt2; t += warp_num_local) {

                        const int p1 = t / nt2;
                        const int p2 = t % nt2;

                        //if ( lane == 0 ) printf("computing %d-%d\n", p1, p2 );

                        auto      o1   = g1.octile[O1 + p1];
                        auto      o2   = g2.octile[O2 + p2];
                        const int nnz1 = __popcll(o1.nzmask);
                        const int nnz2 = __popcll(o2.nzmask);
                        const int I1   = o1.upper;
                        const int J1   = o1.left;
                        const int I2   = o2.upper;
                        const int J2   = o2.left;

                        octile octile1 {p_shared + p1 * shmem_bytes_per_warp + octilex.size_bytes };
                        octile octile2 {p_shared + p2 * shmem_bytes_per_warp + octilex.size_bytes + octile::size_bytes + nzlist::size_bytes};
                        rhs    rhs     {p_shared + warp_id_local * shmem_bytes_per_warp + octilex.size_bytes + octile::size_bytes * 2 + nzlist::size_bytes * 2};

                        // load RHS
                        int j1 = lane / octile_w;
                        int j2 = lane % octile_w;
                        rhs (j1,                        j2) = scratch.p ((J1 + j1                       ) * n2 + (J2 + j2));
                        rhs (j1 + warp_size / octile_w, j2) = scratch.p ((J1 + j1 + warp_size / octile_w) * n2 + (J2 + j2));

                        if (nnz1 * nnz2 >= 256) {
                            float sum_upper = 0, sum_lower = 0;

                            for (int j1 = 0; j1 < octile_w && j1 < g1.n_node - J1; ++j1) {
                                auto e1_upper = octile1 (i1_upper, j1);
                                auto e1_lower = octile1 (i1_lower, j1);
                                auto m1_upper = 1ULL << (i1_upper + j1 * octile_h);
                                auto m1_lower = 1ULL << (i1_lower + j1 * octile_h);
                    
                                #pragma unroll (octile_w)
                                for (int j2 = 0, mask = 1; j2 < octile_w; ++j2, mask <<= 1) {
                                    auto e2 = octile2 (i2, j2);
                                    auto r  = rhs (j1, j2);
                                    auto m2 = 1ULL << (i2 + j2 * octile_h);
                                    if ((o1.nzmask & m1_upper) && (o2.nzmask & m2)) {
                                        sum_upper -= EdgeKernel::compute(e1_upper, e2) * r;
                                    }
                                    if ((o1.nzmask & m1_lower) && (o2.nzmask & m2)) {
                                        sum_lower -= EdgeKernel::compute(e1_lower, e2) * r ;
                                    }
                                }
                            }

                            // printf("threadIdx %d sum_upper  %d %f sum_lower %d %f\n", threadIdx.x, (I1 + i1_upper) * n2 + (I2 + i2), sum_upper, (I1 + i1_lower) * n2 + (I2 + i2), sum_lower);
                            atomicAdd(&scratch.Ap((I1 + i1_upper) * n2 + (I2 + i2)), sum_upper);
                            atomicAdd(&scratch.Ap((I1 + i1_lower) * n2 + (I2 + i2)), sum_lower);
                        } else {
                            nzlist nzlist1 {p_shared + p1 * shmem_bytes_per_warp + octilex.size_bytes + octile1.size_bytes};
                            nzlist nzlist2 {p_shared + p2 * shmem_bytes_per_warp + octilex.size_bytes + octile1.size_bytes + nzlist1.size_bytes + octile2.size_bytes};

                            for (int i = lane; i < nnz1 * nnz2; i += warp_size) {
                                int  k1 = i / nnz2;
                                int  k2 = i - k1 * nnz2;
                                int  p1 = nzlist1(k1);
                                int  p2 = nzlist2(k2);
                                int  i1 = p1 % octile_h;
                                int  j1 = p1 / octile_h;
                                int  i2 = p2 % octile_h;
                                int  j2 = p2 / octile_h;
                                auto e1 = octile1(p1);
                                auto e2 = octile2(p2);
                                auto r  = rhs(j1, j2);
                                atomicAdd(&scratch.Ap((I1 + i1) * n2 + (I2 + i2)), -EdgeKernel::compute(e1, e2) * r);
                            }
                        }
                    }

                    __syncthreads();
                }
            }

            __syncthreads();

            // alpha = rTr / dot( p, Ap );
            auto pAp   = block_vdotv(scratch.p(), scratch.Ap(), N);
            auto alpha = rTz / pAp;

            // x = x + alpha * p;
            // r = r - alpha * Ap;
            // z = M^-1 . r
            // rTr      = r^T . r
            // rTz_next = r^T . z
            float rTr = 0, rTz_next = 0;
            for (int i = threadIdx.x; i < N; i += blockDim.x) {
                scratch.x(i) += alpha * scratch.p(i);
                scratch.r(i) -= alpha * scratch.Ap(i);
                int i1 = i / n2;
                int i2 = i % n2;
                if (i1 < g1.n_node && i2 < g2.n_node) {
                    float d1 = g1.degree[i1] / (1 - q);
                    float d2 = g2.degree[i2] / (1 - q);
                    float D  = d1 * d2;
                    float V  = NodeKernel::compute(g1.node[i1], g2.node[i2]);
                    scratch.z(i) = scratch.r(i) / (D / V);
                }
                rTr += scratch.r(i) * scratch.r(i);
                rTz_next += scratch.r(i) * scratch.z(i);
            }
            __shared__ float sum1, sum2;
            if (threadIdx.x == 0) sum1 = 0, sum2 = 0;
            #pragma unroll
            for (int p = (warp_size >> 1); p >= 1; p >>= 1) {
                rTr      += __shfl_xor_sync(0xFFFFFFFF, rTr, p);
                rTz_next += __shfl_xor_sync(0xFFFFFFFF, rTz_next, p);
            }
            __syncthreads();
            if (laneid() == 0) {
                atomicAdd(&sum1, rTr);
                atomicAdd(&sum2, rTz_next);
            }
            __syncthreads();
            rTr      = sum1;
            rTz_next = sum2;

            // rTr = block_sum(rTr);
            // rTz_next = block_sum(rTz_next);

            #if 0
            __syncthreads();
            if ( threadIdx.x == 0 && blockIdx.x == 0 ) {
                for ( int ij = 0; ij < N; ++ij ) {
                    printf( "iteration %d solution x[%d] = %.7f\n", k, ij, scratch.x( ij ) );
                }
                // printf("rTr %e\n", rTr);
            }
            #endif

            if (rTr < 1e-20f * N * N) break;

            auto beta = rTz_next / rTz;

            // p = r + beta * p;
            for (int i = threadIdx.x; i < N; i += blockDim.x) {
                //scratch.p(i) = scratch.r(i) + beta * scratch.p(i);
                int i1 = i / n2;
                int i2 = i % n2;
                if (i1 < g1.n_node && i2 < g2.n_node) {
                    float p       = scratch.z(i) + beta * scratch.p(i);
                    float d1      = g1.degree[i1] / (1 - q);
                    float d2      = g2.degree[i2] / (1 - q);
                    scratch.p(i)  = p;
                    scratch.Ap(i) = d1 * d2 / NodeKernel::compute(g1.node[i1], g2.node[i2]) * p;
                }
            }
            __syncthreads();

            rTz = rTz_next;
        }
        __syncthreads();

        for (int i = threadIdx.x; i < N; i += blockDim.x) {
            int i1 = i / n2;
            int i2 = i % n2;
            if (i1 < g1.n_node && i2 < g2.n_node) {
                auto r = scratch.x(i);
                if (lmin == 1) {
                    r -=  NodeKernel::compute(g1.node[i1], g2.node[i2]) * q * q / (q0 * q0);
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

}  // namespace marginalized

}  // namespace graphdot

#endif
