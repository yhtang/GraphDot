#include <array.h>
#include <basekernel.h>
#include <graph.h>
#include <marginalized_kernel.h>
#include <fmath.h>
#include <frozen_array.h>
#include <numpy_type.h>
#include <util_cuda.h>

using namespace graphdot::numpy_type;
using namespace graphdot::basekernel;
namespace solver_ns = graphdot::marginalized;

${node_kernel}
${edge_kernel}
${p_start}

using node_t = ${node_t};
using edge_t = ${edge_t};

using graph_t   = graphdot::graph_t<node_t, edge_t>;
using scratch_t = solver_ns::pcg_scratch_t;
using solver_t  = solver_ns::labeled_compact_block_dynsched_pcg<graph_t>;

__constant__ char shmem_bytes_per_warp[solver_t::shmem_bytes_per_warp];

extern "C" {
    __global__ void graph_kernel_solver(
        graph_t const   * graphs,
        scratch_t       * scratches,
        uint2           * jobs,
        uint32          * starts,
        float32         * out,
        uint32          * i_job_global,
        const uint32      n_jobs,
        const uint32      out_h,
        const uint32      out_w,
        const float32     q,
        const float32     q0
    ) {
        extern __shared__ char shmem[];
        __shared__ uint32 i_job;

        const int lane = graphdot::cuda::laneid();
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

            // wipe output buffer for atomic accumulations
            if (?{traits.nodal is False}) {
                if (?{traits.diagonal is True}) {
                    out[I1] = 0.f;
                } else {
                    out[I1 + I2 * out_h] = 0.f;
                    if (?{traits.symmetric is True}) {
                        if (job.x != job.y) out[I2 + I1 * out_h] = 0.f;
                    }
                }
            }

            if (?{traits.eval_gradient is True}) {
                solver_t::compute_duo(
                    node_kernel,
                    edge_kernel,
                    p_start,
                    g1, g2,
                    scratch,
                    shmem,
                    q, q0);
            } else {
                solver_t::compute(
                    node_kernel,
                    edge_kernel,
                    g1, g2,
                    scratch,
                    shmem,
                    q, q0);
            }
            __syncthreads();

            /********* post-processing *********/

            // apply starting probability and min-path truncation
            if (?{traits.lmin == 1}) {
                for (int i = threadIdx.x; i < N; i += blockDim.x) {
                    int i1 = i / n2;
                    int i2 = i % n2;
                    scratch.x(i) -= node_kernel(g1.node[i1], g2.node[i2]) * q * q / (q0 * q0);
                }
            }
            __syncthreads();

            // write to output buffer
            if (?{traits.nodal == "block"}) {
                for (int i = threadIdx.x; i < N; i += blockDim.x) {
                    int i1 = i / n2;
                    int i2 = i % n2;
                    out[I1 + i1 + i2 * g1.n_node] =
                        scratch.x(i) * p_start(g1.node[i1]) * p_start(g2.node[i2]);
                }
            }
            if (?{traits.nodal is True}) {
                if (?{traits.diagonal is True}) {
                    for (int i1 = threadIdx.x; i1 < g1.n_node; i1 += blockDim.x) {
                        out[I1 + i1] = scratch.x(i1 + i1 * n1) * graphdot::ipow<2>(p_start(g1.node[i1]));
                    }
                } else {
                    for (int i = threadIdx.x; i < N; i += blockDim.x) {
                        int i1 = i / n2;
                        int i2 = i % n2;
                        auto r = scratch.x(i) * p_start(g1.node[i1]) * p_start(g2.node[i2]);
                        out[(I1 + i1) + (I2 + i2) * out_h] = r;
                        if (?{traits.symmetric is True}) {
                            if (job.x != job.y) {
                                out[(I2 + i2) + (I1 + i1) * out_h] = r;
                            }
                        }
                    }    
                }
            }
            if (?{traits.nodal is False}) {
                float32 sum = 0;
                for (int i = threadIdx.x; i < N; i += blockDim.x) {
                    int i1 = i / n2;
                    int i2 = i % n2;
                    sum += scratch.x(i) * p_start(g1.node[i1]) * p_start(g2.node[i2]);
                }
                sum = graphdot::cuda::warp_sum(sum);
                if (lane == 0) {
                    if (?{traits.diagonal is True}) {
                        atomicAdd(out + I1, sum);
                    } else {
                        atomicAdd(out + I1 + I2 * out_h, sum);
                        if (?{traits.symmetric is True}) {
                            if (job.x != job.y) {
                                atomicAdd(out + I2 + I1 * out_h, sum);
                            }
                        }
                    }
                }

                if (?{traits.eval_gradient is True}) {

                    auto jacobian = solver_t::derivative(
                        p_start,
                        node_kernel,
                        edge_kernel,
                        g1, g2,
                        scratch,
                        shmem,
                        q
                    );

                    if (?{traits.diagonal is True}) {
                        for (int i = threadIdx.x; i < jacobian.size; i += blockDim.x) {
                            out[I1 + (i + 1) * out_h] = 0;
                        }
                        __syncthreads();

                        #pragma unroll (jacobian.size)
                        for(int i = 0; i < jacobian.size; ++i) {
                            auto j = graphdot::cuda::warp_sum(jacobian[i]);
                            if (lane == 0) {
                                atomicAdd(out + I1 + (i + 1) * out_h, j);
                            };
                        }
                        __syncthreads();
                    } else {
                        for (int i = threadIdx.x; i < jacobian.size; i += blockDim.x) {
                            out[I1 + I2 * out_h + (i + 1) * out_h * out_w] = 0;
                            if (?{traits.symmetric is True}) {
                                if (job.x != job.y) {
                                    out[I2 + I1 * out_h + (i + 1) * out_h * out_w] = 0;
                                }
                            }
                        }
                        __syncthreads();

                        #pragma unroll (jacobian.size)
                        for(int i = 0; i < jacobian.size; ++i) {
                            auto j = graphdot::cuda::warp_sum(jacobian[i]);
                            if (lane == 0) {
                                atomicAdd(out + I1 + I2 * out_h + (i + 1) * out_h * out_w, j);
                                if (?{traits.symmetric is True}) {
                                    if (job.x != job.y) {
                                        atomicAdd(out + I2 + I1 * out_h + (i + 1) * out_h * out_w, j);
                                    }
                                }
                            };
                        }
                        __syncthreads();
                    }
                }
            }
        }
    }
}
