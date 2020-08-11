#include <array.h>
#include <basekernel.h>
#include <fmath.h>
#include <frozen_array.h>
#include <graph.h>
#include <marginalized_kernel.h>
#include <numpy_type.h>
#include <tensor_view.h>
#include <util_cuda.h>

using namespace graphdot::cuda;
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
        float32        ** diags,
        scratch_t       * scratches,
        uint2           * jobs,
        uint            * starts,
        float32         * gramian,
        float32         * gradient,
        uint            * i_job_global,
        const uint        n_jobs,
        const uint        nX,
        const uint        nY,
        const uint        nJ,
        const float32     q,
        const float32     q0
    ) {
        extern __shared__ char shmem[];
        __shared__ uint i_job;

        auto scratch = scratches[blockIdx.x];

        while (true) {
            if (threadIdx.x == 0) i_job = atomicInc(i_job_global, 0xFFFFFFFF);
            __syncthreads();

            if (i_job >= n_jobs) break;

            const auto job = jobs[i_job];
            const auto g1  = graphs[job.x];
            const auto g2  = graphs[job.y];
            const auto I1  = starts[job.x];
            const auto I2  = starts[job.y];
            const uint n1  = g1.n_node;
            const uint n2  = g2.n_node;
            const uint N   = n1 * n2;
            auto const diag1 = diags[job.x];
            auto const diag2 = diags[job.y];

            // setup metric matrix view
            auto K = graphdot::tensor_view(gramian, nX, nY);

            // setup Jacobian matrix view
            #if ?{traits.eval_gradient is True}
                auto J = graphdot::tensor_view(gradient, nX, nY, nJ);
            #endif

            // solve the MLGK equation
            #if ?{traits.eval_gradient is True}
                solver_t::compute_duo(
                    node_kernel,
                    edge_kernel,
                    p_start,
                    g1, g2,
                    scratch,
                    shmem,
                    q, q0);
            #else
                solver_t::compute(
                    node_kernel,
                    edge_kernel,
                    g1, g2,
                    scratch,
                    shmem,
                    q, q0);
            #endif
            __syncthreads();

            // reusing r for d12 and z for d21
            auto d12 = scratch.r();
            auto d21 = scratch.z();
            for(int i = threadIdx.x; i < max(n1, n2); i += blockDim.x) {
                d12[i] = std::numeric_limits<float32>::max();
                d21[i] = std::numeric_limits<float32>::max();
            }
            __syncthreads();

            for (int i = threadIdx.x; i < N; i += blockDim.x) {
                int i1 = i / n2;
                int i2 = i % n2;
                const auto x = scratch.x(i);
                // apply min-path truncation
                #if ?{traits.lmin == 1}
                    x -= node_kernel(g1.node[i1], g2.node[i2]) * q * q / (q0 * q0);
                #endif
                auto r12 = x * p_start(g1.node[i1]) * p_start(g2.node[i2]);
                auto r1 = diag1[i1];
                auto r2 = diag2[i2];
                auto k = r12 * rsqrtf(r1 * r2);
                auto d = sqrtf(max(0.f, 2 - 2 * k));
                atomicMin(d12 + i1, d);
                atomicMin(d21 + i2, d);
            }
            __syncthreads();

            for (int i = threadIdx.x; i < n1; i += blockDim.x) {
                atomicMax(d12, d12[i]);
            }
            for (int i = threadIdx.x; i < n2; i += blockDim.x) {
                atomicMax(d21, d21[i]);
            }
            __syncthreads();

            // write to output buffer
            auto dh = max(*d12, *d21);
            if (threadIdx.x == 0) {
                K(I1, I2) = dh;
                if (?{traits.symmetric is True}) {
                    if (job.x != job.y) {
                        K(I2, I1) = dh;
                    }
                }
            }
            __syncthreads();

            #if ?{traits.eval_gradient is True}
            // TBD
            #endif
        }
    }
}
