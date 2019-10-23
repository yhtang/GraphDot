#include <graph.h>
#include <marginalized_kernel.h>
#include <numpy_type.h>

//using namespace graphdot;
using namespace graphdot::numpy_type;

${node_kernel}
${edge_kernel}
using node_t = ${node_t};
using edge_t = ${edge_t};

using graph_t   = graphdot::graph_t<node_t, edge_t>;
using scratch_t = graphdot::marginalized::pcg_scratch_t;
using job_t     = graphdot::marginalized::job_t;
using solver_t  = graphdot::marginalized::labeled_compact_block_dynsched_pcg<graph_t>;

__constant__ char shmem_bytes_per_warp[solver_t::shmem_bytes_per_warp];

extern "C" {
    __global__ void graph_kernel_solver(
        graph_t const * graphs,
        scratch_t     * scratch,
        job_t         * jobs,
        unsigned int  * i_job_global,
        const unsigned  n_jobs,
        const float     q,
        const float     q0,
        const int       lmin
    ) {
        extern __shared__ char shmem[];
        __shared__ unsigned int i_job;

        while (true) {
            if (threadIdx.x == 0) i_job = atomicInc(i_job_global, 0xFFFFFFFF);
            __syncthreads();

            if (i_job >= n_jobs) break;

            solver_t::compute<node_kernel, edge_kernel> (
                graphs[ jobs[i_job].i ],
                graphs[ jobs[i_job].j ],
                scratch[ blockIdx.x ],
                shmem, q, q0, lmin,
                jobs[i_job].vr);
            __syncthreads();
        }
    }
}
