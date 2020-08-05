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
        float32         * gramian,
        float32         * gradient,
        uint32          * i_job_global,
        const uint32      n_jobs,
        const uint32      gramian_h,
        const uint32      gramian_w,
        const uint32      gradient_dim,
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
                    gramian[I1 + i1 + i2 * g1.n_node] =
                        scratch.x(i) * p_start(g1.node[i1]) * p_start(g2.node[i2]);
                }
            }
            if (?{traits.nodal is True}) {
                if (?{traits.diagonal is True}) {
                    for (int i1 = threadIdx.x; i1 < g1.n_node; i1 += blockDim.x) {
                        gramian[I1 + i1] = scratch.x(i1 + i1 * n1) * graphdot::ipow<2>(p_start(g1.node[i1]));
                    }
                } else {
                    for (int i = threadIdx.x; i < N; i += blockDim.x) {
                        int i1 = i / n2;
                        int i2 = i % n2;
                        auto r = scratch.x(i) * p_start(g1.node[i1]) * p_start(g2.node[i2]);
                        gramian[(I1 + i1) + (I2 + i2) * gramian_h] = r;
                        if (?{traits.symmetric is True}) {
                            if (job.x != job.y) {
                                gramian[(I2 + i2) + (I1 + i1) * gramian_h] = r;
                            }
                        }
                    }    
                }
            }
            if (?{traits.nodal is False}) {
                // wipe output buffer for atomic accumulations
                if (threadIdx.x == 0) {
                    if (?{traits.diagonal is True}) {
                        gramian[I1] = 0.f;
                   } else {
                       gramian[I1 + I2 * gramian_h] = 0.f;
                       if (?{traits.symmetric is True}) gramian[I2 + I1 * gramian_h] = 0.f;
                   }   
                }

                __syncthreads();

                float32 sum = 0;
                for (int i = threadIdx.x; i < N; i += blockDim.x) {
                    int i1 = i / n2;
                    int i2 = i % n2;
                    sum += scratch.x(i) * p_start(g1.node[i1]) * p_start(g2.node[i2]);
                }
                sum = graphdot::cuda::warp_sum(sum);
                if (lane == 0) {
                    if (?{traits.diagonal is True}) {
                        atomicAdd(gramian + I1, sum);
                    } else {
                        atomicAdd(gramian + I1 + I2 * gramian_h, sum);
                        if (?{traits.symmetric is True}) {
                            if (job.x != job.y) {
                                atomicAdd(gramian + I2 + I1 * gramian_h, sum);
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

                    // wipe output buffer for atomic accumulations
                    if (?{traits.diagonal is True}) {
                        for (int i = threadIdx.x; i < jacobian.size; i += blockDim.x) {
                            gradient[i + I1 * gradient_dim] = 0;
                        }    
                    } else {
                        for (int i = threadIdx.x; i < jacobian.size; i += blockDim.x) {
                            gradient[i + I1 * gradient_dim + I2 * gradient_dim * gramian_h] = 0;
                            if (?{traits.symmetric is True} && job.x != job.y) {
                                gradient[i + I1 * gradient_dim + I2 * gradient_dim * gramian_h] = 0;
                            }
                        }
                    }

                    if (?{traits.diagonal is True}) {
                        #pragma unroll (jacobian.size)
                        for(int i = 0; i < jacobian.size; ++i) {
                            auto j = graphdot::cuda::warp_sum(jacobian[i]);
                            if (lane == 0) {
                                atomicAdd(gradient + i + I1 * gradient_dim, j);
                            };
                        }
                        __syncthreads();
                    } else {
                        #pragma unroll (jacobian.size)
                        for(int i = 0; i < jacobian.size; ++i) {
                            auto j = graphdot::cuda::warp_sum(jacobian[i]);
                            if (lane == 0) {
                                atomicAdd(gradient + i + I1 * gradient_dim + I2 * gradient_dim * gramian_h, j);
                                if (?{traits.symmetric is True}) {
                                    if (job.x != job.y) {
                                        atomicAdd(gradient + i + I2 * gradient_dim + I1 * gradient_dim * gramian_h, j);
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
