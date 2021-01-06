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

namespace num_hacks {
    // Use 0.9999995 instead of 1 to cancel some round-off error
    // and ensure self-distances are close enough to 0.
    constexpr static float32 one = 0.9999995f;
    // Use eps to nudge the reciprocal of distance so that gradients near
    // 0 do not explode.
    constexpr static float32 eps = 0.0001f;    
}


extern "C" {
    __global__ void graph_maximin_distance(
        graph_t const   * graphs,
        float32        ** diags,
        scratch_t       * scratch_pcg,
        uint2           * jobs,
        uint            * starts,
        float32         * gramian,
        int32           * active,
        float32         * gradient,
        uint            * i_job_global,
        const uint        n_jobs,
        const uint        nX,
        const uint        nY,
        const uint        nJ,
        const float32     q,
        const float32     q0,
        const float32     eps_diff,
        const float32     ftol,
        const float32     gtol
    ) {
        extern __shared__ char shmem[];
        __shared__ uint i_job;

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
            const auto I1  = starts[job.x];
            const auto I2  = starts[job.y];
            const uint n1  = g1.n_node;
            const uint n2  = g2.n_node;
            const uint N   = n1 * n2;
            auto const diag1 = diags[job.x];
            auto const diag2 = diags[job.y];

            //------------------------------------------------------------------
            // I. Evaluate the graph kernel between two graphs.
            //------------------------------------------------------------------

            // I.1. Set up view to the distance matrix output buffer.
            // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            auto K = graphdot::tensor_view(gramian, nX, nY);
            auto A = graphdot::tensor_view(active, nX, nY);

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

            // I.3.1. Zero-out reduction buffers for maxi-mini operations.
            auto d12 = scratch.ext(0);
            auto d21 = scratch.ext(1);
            auto pairwise_dist = scratch.ext(2);
            A(I1, I2) = 0;
            if (?{traits.symmetric is True} && job.x != job.y) {
                A(I2, I1) = 0;
            }
            for(int i = threadIdx.x; i < max(n1, n2); i += blockDim.x) {
                d12[i] = std::numeric_limits<float32>::max();
                d21[i] = std::numeric_limits<float32>::max();
            }
            __syncthreads();

            auto const postproc = [&](auto x, auto i1, auto i2){
                #if ?{traits.lmin == 1}
                    x -= node_kernel(g1.node[i1], g2.node[i2]) * q * q / (q0 * q0);
                #endif
                return x * p_start(g1.node[i1]) * p_start(g2.node[i2]);
            };

            auto const kernel_induced_distance = [](auto k12, auto k1, auto k2){
                return sqrtf(max(0.f, num_hacks::one - k12 * rsqrtf(k1 * k2)));
            };

            auto const normalized_kernel_grad = [](
                auto k12, auto dk12, auto k1, auto dk1, auto k2, auto dk2
            ){
                return dk12 * rsqrtf(k1 * k2) - 0.5f * k12 * rsqrtf(graphdot::ipow<3>(k1 * k2)) * (dk1 * k2 + k1 * dk2);
            };

            // I.3.2. Compute kernel-induced pairwise distances and perform
            //        minimum reductions along rows follwoed by maximum
            //        reductions along columns
            for (int i = threadIdx.x; i < N; i += blockDim.x) {
                int i1 = i / n2;
                int i2 = i % n2;
                auto k12 = postproc(scratch.x(i), i1, i2);
                // auto dsqr = kernel_square_distance(k12, diag1[i1], diag2[i2]);
                auto d = kernel_induced_distance(k12, diag1[i1], diag2[i2]);
                pairwise_dist[i] = d;
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

            // I.3.3. Write to output buffer.
            auto dh = max(*d12, *d21);
            if (threadIdx.x == 0) {
                K(I1, I2) = dh;
                if (?{traits.symmetric is True} && job.x != job.y) {
                    K(I2, I1) = dh;
                }
            }
            for(int i = threadIdx.x; i < N; i += blockDim.x) {
                if (pairwise_dist[i] == dh) {
                    atomicMax(A.at(I1, I2), i);
                    if (?{traits.symmetric is True} && job.x != job.y) {
                        int i1 = i / n2;
                        int i2 = i % n2;        
                        atomicMax(A.at(I2, I1), i2 * n1 + i1);
                    }    
                }
            }
            __syncthreads();

            //------------------------------------------------------------------
            // II. Evaluate the gradient of the maximin distance with respect to
            //     hyperparameters
            //------------------------------------------------------------------

            #if ?{traits.eval_gradient is True}

                // II.1. Set up views to output buffer and temporaries
                // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                auto J = graphdot::tensor_view(gradient, nX, nY, nJ);
                auto const x0 = scratch.ext(0);
                auto const diff = scratch.ext(1);

                // II.2. Save the raw solution as initial guess for finite
                //        difference scheme
                // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                for (int i = threadIdx.x; i < N; i += blockDim.x) {
                    x0[i] = scratch.x(i);
                }
                __syncthreads();

                // II.3. Calculate the gradients and save to the output.
                // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                auto const active_i = A(I1, I2);

                constexpr static int _offset_p = 0;
                constexpr static int _offset_q = _offset_p + p_start.jac_dims;
                constexpr static int _offset_v = _offset_q + 1;
                constexpr static int _offset_e = _offset_v + node_kernel.jac_dims;

                // II.3.1 Except for p, calculate the gradients of the graph
                //        kernel first on the 'hotspot' node pair.

                // dp must be done first, otherwise scratch.x will be wiped
                // by dq, dv, and de.

                // === dK/dp ===

                if (threadIdx.x == 0) {
                    int i1 = active_i / n2;
                    int i2 = active_i % n2;                    
                    auto k1 = diag1[i1];
                    auto k2 = diag2[i2];
                    auto k12 = postproc(scratch.x(active_i), i1, i2);
                    auto d = kernel_induced_distance(k12, k1, k2);
                    auto dk1 = graphdot::tensor_view(diag1 + n1, n1, nJ);
                    auto dk2 = graphdot::tensor_view(diag2 + n2, n2, nJ);
                    auto p1 = p_start(g1.node[i1]);
                    auto p2 = p_start(g2.node[i2]);
                    auto dp1 = p_start._j_a_c_o_b_i_a_n_(g1.node[i1]);
                    auto dp2 = p_start._j_a_c_o_b_i_a_n_(g2.node[i2]);
                    auto x = scratch.x(active_i);
                    #if ?{traits.lmin == 1}
                        x -= node_kernel(g1.node[i1], g2.node[i2]) * q * q / (q0 * q0);
                    #endif

                    for(int j = 0; j < p_start.jac_dims; ++j) {
                        auto dk12 = x * (p1 * dp2[j] + p2 * dp1[j]);
                        auto dk = normalized_kernel_grad(k12, dk12, k1, dk1(i1, _offset_p + j), k2, dk2(i2, _offset_p + j));
                        auto grad = -0.5f * dk / (d + num_hacks::eps);
                        J(I1, I2, _offset_p + j) = grad;
                        #if ?{traits.symmetric is True}
                            if (job.x != job.y) J(I2, I1, _offset_p + j) = grad;
                        #endif
                    }
                }

                // === dK/dq ===

                for (int i = threadIdx.x; i < N; i += blockDim.x) {
                    scratch.x(i) = x0[i];
                }
                __syncthreads();

                solver_t::compute(
                    node_kernel,
                    edge_kernel,
                    g1, g2,
                    scratch,
                    shmem,
                    expf(logf(q) + eps_diff), expf(logf(q0) + eps_diff),
                    true,
                    gtol);
                __syncthreads();

                for (int i = threadIdx.x; i < N; i += blockDim.x) {
                    diff[i] = scratch.x(i);
                    scratch.x(i) = x0[i];
                }
                __syncthreads();

                solver_t::compute(
                    node_kernel,
                    edge_kernel,
                    g1, g2,
                    scratch,
                    shmem,
                    expf(logf(q) - eps_diff), expf(logf(q0) - eps_diff),
                    true,
                    gtol);
                __syncthreads();

                if (threadIdx.x == 0) {
                    J(I1, I2, _offset_q) = (diff[active_i] - scratch.x(active_i)) / (2 * eps_diff * q);
                }                    
                __syncthreads();

                // === dK/dv ===

                for(int j = 0; j < node_kernel.jac_dims; ++j) {
                    auto const diff = scratch.ext(1);

                    for (int i = threadIdx.x; i < N; i += blockDim.x) {
                        scratch.x(i) = x0[i];
                    }
                    __syncthreads();

                    solver_t::compute(
                        node_kernel_diff_grid[j * 2],
                        edge_kernel,
                        g1, g2,
                        scratch,
                        shmem,
                        q, q0,
                        true,
                        gtol);
                    __syncthreads();

                    for (int i = threadIdx.x; i < N; i += blockDim.x) {
                        diff[i] = scratch.x(i);
                        scratch.x(i) = x0[i];
                    }
                    __syncthreads();

                    solver_t::compute(
                        node_kernel_diff_grid[j * 2 + 1],
                        edge_kernel,
                        g1, g2,
                        scratch,
                        shmem,
                        q, q0,
                        true,
                        gtol);
                    __syncthreads();
        
                    if (threadIdx.x == 0) {
                        J(I1, I2, _offset_v + j) = (diff[active_i] - scratch.x(active_i)) / (2 * eps_diff * node_kernel_flat_theta[j]);
                    }                    
                    __syncthreads();
                }

                // === dK/de ===

                for(int j = 0; j < edge_kernel.jac_dims; ++j) {
                    auto const diff = scratch.ext(1);

                    for (int i = threadIdx.x; i < N; i += blockDim.x) {
                        scratch.x(i) = x0[i];
                    }
                    __syncthreads();

                    solver_t::compute(
                        node_kernel,
                        edge_kernel_diff_grid[j * 2],
                        g1, g2,
                        scratch,
                        shmem,
                        q, q0,
                        true,
                        gtol);
                    __syncthreads();

                    for (int i = threadIdx.x; i < N; i += blockDim.x) {
                        diff[i] = scratch.x(i);
                        scratch.x(i) = x0[i];
                    }
                    __syncthreads();

                    solver_t::compute(
                        node_kernel,
                        edge_kernel_diff_grid[j * 2 + 1],
                        g1, g2,
                        scratch,
                        shmem,
                        q, q0,
                        true,
                        gtol);
                    __syncthreads();

                    if (threadIdx.x == 0) {
                        J(I1, I2, _offset_e + j) = (diff[active_i] - scratch.x(active_i)) / (2 * eps_diff * edge_kernel_flat_theta[j]);
                    }                    
                    __syncthreads();
                }

                // II.3.2 Calculate the gradients of the distance from those of
                //        the kernel.

                if (threadIdx.x == 0) {
                    int i1 = active_i / n2;
                    int i2 = active_i % n2;
                    auto k12 = postproc(scratch.x(active_i), i1, i2);
                    auto k1 = diag1[i1];
                    auto k2 = diag2[i2];
                    auto d = kernel_induced_distance(k12, k1, k2);
                    auto dk1 = graphdot::tensor_view(diag1 + n1, n1, nJ);
                    auto dk2 = graphdot::tensor_view(diag2 + n2, n2, nJ);
                    auto p1 = p_start(g1.node[i1]);
                    auto p2 = p_start(g2.node[i2]);
                    for(int j = _offset_q; j < nJ; ++j) {
                        auto dk12 = J(I1, I2, j) * p1 * p2;
                        auto dk = normalized_kernel_grad(k12, dk12, k1, dk1(i1, j), k2, dk2(i2, j));
                        auto grad = -0.5f * dk / (d + num_hacks::eps);
                        J(I1, I2, j) = grad;
                        if (?{traits.symmetric is True} && job.x != job.y) {
                            J(I2, I1, j) = grad;
                        }
                    }
                }

            #endif
        }
    }
}
