#include <array.h>
#include <basekernel.h>
#include <fmath.h>
#include <frozen_array.h>
#include <graph.h>
#include <marginalized_kernel.h>
#include <numpy_type.h>
#include <tensor_view.h>
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
    __global__ void alt_graph_kernel_solver(
        graph_t const   * graphs,
        scratch_t       * scratch_pcg,
        uint2           * jobs,
        float32         * gramian,
        uint            * i_job_global,
        const uint        n_jobs,
        const float32     q,
        const float32     q0,
        const float32     ftol,
        const float32     gtol
    ) {
        extern __shared__ char shmem[];
        __shared__ uint i_job;

        const int lane = graphdot::cuda::laneid();
        auto scratch = scratch_pcg[blockIdx.x];

        //======================================================================
        // OUTER LOOP over pairs of graphs.
        //======================================================================

        while (true) {
            if (threadIdx.x == 0) i_job = atomicInc(i_job_global, 0xFFFFFFFF);
            __syncthreads();

            if (i_job >= n_jobs) break;

            const auto job = jobs[i_job];
            const auto g1  = graphs[job.x];
            const auto g2  = graphs[job.y];
            const auto I   = i_job;
            const uint n1  = g1.n_node;
            const uint n2  = g2.n_node;
            const uint N   = n1 * n2;

            //------------------------------------------------------------------
            // I. Evaluate the graph kernel between two graphs.
            //------------------------------------------------------------------

            // I.1. Set up view to the kernel matrix output buffer.
            // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            auto K = graphdot::tensor_view(gramian, n_jobs);

            // I.2. Solve the primary MLGK equation.
            // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            solver_t::compute(
                node_kernel,
                edge_kernel,
                g1, g2,
                scratch,
                shmem,
                q, q0,
                false,
                ftol);
            __syncthreads();

            // I.3. Postprocessing
            // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            auto const postproc = [&](auto *x){
                #if ?{traits.lmin == 1}
                    for (int i = threadIdx.x; i < N; i += blockDim.x) {
                        int i1 = i / n2;
                        int i2 = i % n2;
                        x[i] -= node_kernel(g1.node[i1], g2.node[i2]) * q * q / (q0 * q0);
                    }
                #endif
            };
            postproc(scratch.x());
            __syncthreads();

            // I.4. write kernel matrix elements to output
            // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            // wipe output buffer for atomic accumulations
            if (threadIdx.x == 0) {
                K(I) = 0.f;
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
                atomicAdd(K.at(I), sum);
            }
        }
    }
}
