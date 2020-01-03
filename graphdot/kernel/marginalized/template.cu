#include <graph.h>
#include <marginalized_kernel.h>
#include <fmath.h>
#include <numpy_type.h>
#include <util_cuda.h>

using namespace graphdot::numpy_type;

${node_kernel}
${edge_kernel}

using node_t = ${node_t};
using edge_t = ${edge_t};

using graph_t   = graphdot::graph_t<node_t, edge_t>;
using scratch_t = graphdot::marginalized::pcg_scratch_t;
using solver_t  = graphdot::marginalized::labeled_compact_block_dynsched_pcg<graph_t>;

__constant__ char shmem_bytes_per_warp[solver_t::shmem_bytes_per_warp];

extern "C" {
    __global__ void graph_kernel_solver(
        graph_t const   * graphs,
        float32        ** p,
        scratch_t       * scratches,
        uint2           * jobs,
        uint32          * starts,
        float32         * R,
        uint32          * i_job_global,
        const uint32      n_jobs,
        const uint32      R_stride,
        const float32     q,
        const float32     q0
    ) {
        extern __shared__ char shmem[];
        __shared__ uint32 i_job;

        auto scratch = scratches[blockIdx.x];

        while (true) {
            if (threadIdx.x == 0) i_job = atomicInc(i_job_global, 0xFFFFFFFF);
            __syncthreads();

            if (i_job >= n_jobs) break;

            auto const job = jobs[i_job];
            auto const g1  = graphs[job.x];
            auto const g2  = graphs[job.y];
            auto const I1  = std::size_t(starts[job.x]);
            auto const I2  = std::size_t(starts[job.y]);
            const int  n1  = g1.n_node;
            const int  n2  = g2.n_node;
            const int   N  = n1 * n2;
            auto const p1  = p[job.x];
            auto const p2  = p[job.y];

            // wipe output buffer for atomic accumulations
            ?{traits.nodal is False} {
                ?{traits.diagonal is True} {
                    R[I1] = 0.f;
                } else {
                    R[I1 + I2 * R_stride] = 0.f;
                    ?{traits.symmetric is True} {
                        if (job.x != job.y) R[I2 + I1 * R_stride] = 0.f;
                    }
                }
            }

            solver_t::compute(node_kernel, edge_kernel, g1, g2, scratch, shmem, q, q0);
            __syncthreads();

            /********* post-processing *********/

            // apply starting probability and min-path truncation
            for (int i = threadIdx.x; i < N; i += blockDim.x) {
                int i1 = i / n2;
                int i2 = i % n2;
                auto r = scratch.x(i);
                ?{traits.lmin == 1} {
                    r -= node_kernel(g1.node[i1], g2.node[i2]) * q * q / (q0 * q0);
                }
                scratch.x(i) = r * p1[i1] * p2[i2];
            }

            // write to output buffer
            ?{traits.nodal == "block"} {
                for (int i = threadIdx.x; i < N; i += blockDim.x) {
                    int i1 = i / n2;
                    int i2 = i % n2;
                    auto r = scratch.x(i);
                    R[I1 + i1 + i2 * g1.n_node] = r;
                }
            }
            ?{traits.nodal is True} {
                ?{traits.diagonal is True}{
                    for (int i1 = threadIdx.x; i1 < g1.n_node; i1 += blockDim.x) {
                        R[I1 + i1] = scratch.x(i1 + i1 * n1);
                    }
                } else {
                    for (int i = threadIdx.x; i < N; i += blockDim.x) {
                        int i1 = i / n2;
                        int i2 = i % n2;
                        auto r = scratch.x(i);
                        R[(I1 + i1) + (I2 + i2) * R_stride] = r;
                        ?{traits.symmetric is True} {
                            if (job.x != job.y) R[(I2 + i2) + (I1 + i1) * R_stride] = r;
                        }
                    }    
                }
            }
            ?{traits.nodal is False} {
                float32 sum = 0;
                for (int i = threadIdx.x; i < N; i += blockDim.x) {
                    sum += scratch.x(i);
                }
                sum = graphdot::cuda::warp_sum(sum);
                if (graphdot::cuda::laneid() == 0) {
                    ?{traits.diagonal is True} {
                        atomicAdd(R + I1, sum);
                    } else {
                        atomicAdd(R + I1 + I2 * R_stride, sum);
                        ?{traits.symmetric is True} {
                            if (job.x != job.y) {
                                atomicAdd(R + I2 + I1 * R_stride, sum);
                            }
                        }
                    }
                }
            }
        }
    }
}
