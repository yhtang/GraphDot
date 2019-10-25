#include <graph.h>
#include <marginalized_kernel.h>
#include <numpy_type.h>

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
        graph_t const * graphs,
        scratch_t     * scratches,
        uint2         * jobs,
        uint          * starts,
        float         * R,
        unsigned int  * i_job_global,
        const unsigned  n_jobs,
        const unsigned  R_stride,
        const float     q,
        const float     q0,
        const int       lmin,
        const int       symmetric
    ) {
        extern __shared__ char shmem[];
        __shared__ unsigned int i_job;

        auto scratch = scratches[blockIdx.x];

        while (true) {
            if (threadIdx.x == 0) i_job = atomicInc(i_job_global, 0xFFFFFFFF);
            __syncthreads();

            if (i_job >= n_jobs) break;

            auto const job = jobs[i_job];
            auto const g1  = graphs[job.x];
            auto const g2  = graphs[job.y];

            solver_t::compute<node_kernel, edge_kernel>(g1, g2, scratch, shmem, q, q0);
            __syncthreads();

            // post-processing and write to output matrix
            const int         n1 = g1.padded_size();
            const int         n2 = g2.padded_size();
            const int          N = n1 * n2;
            const std::size_t I1 = starts[job.x];
            const std::size_t I2 = starts[job.y];
            for (int i = threadIdx.x; i < N; i += blockDim.x) {
                int i1 = i / n2;
                int i2 = i % n2;
                if (i1 < g1.n_node && i2 < g2.n_node) {
                    auto r = scratch.x(i);
                    if (lmin == 1) {
                        r -= node_kernel::compute(g1.node[i1], g2.node[i2]) * q * q / (q0 * q0);
                    }
                    R[(I1 + i1) + (I2 + i2) * R_stride] = r;
                    if (symmetric && job.x != job.y) {
                        R[(I2 + i2) + (I1 + i1) * R_stride] = r;
                    }
                }
            }    
        }
    }
}
