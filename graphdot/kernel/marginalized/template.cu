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
    __global__ void graph_kernel_solver(
        graph_t const   * graphs,
        scratch_t       * scratch_pcg,
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
        const float32     q0,
        const float32     eps_diff,
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
            const auto I1  = starts[job.x];
            const auto I2  = starts[job.y];
            const uint n1  = g1.n_node;
            const uint n2  = g2.n_node;
            const uint N   = n1 * n2;

            //------------------------------------------------------------------
            // I. Evaluate the graph kernel between two graphs.
            //------------------------------------------------------------------

            // I.1. Set up view to the kernel matrix output buffer.
            // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            #if ?{traits.nodal == "block"}
                auto K = graphdot::tensor_view(gramian, nX);
            #elif ?{traits.nodal is True}
                #if ?{traits.diagonal is True}
                    auto K = graphdot::tensor_view(gramian, nX);
                #else
                    auto K = graphdot::tensor_view(gramian, nX, nY);
                #endif
            #elif ?{traits.nodal is False}
                #if ?{traits.diagonal is True}
                    auto K = graphdot::tensor_view(gramian, nX);
                #else
                    auto K = graphdot::tensor_view(gramian, nX, nY);
                #endif
            #endif

            // I.2. Solve the primary MLGK equation.
            // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            #if ?{traits.eval_gradient is True and traits.nodal is False}
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
                    q, q0,
                    false,
                    ftol);
            #endif
            __syncthreads();

            // I.3. Postprocessing
            // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

            // I.3.1. Save the raw solution as initial guess for finite
            //        difference scheme
            #if ?{traits.eval_gradient is True and traits.nodal is True}

                auto const x0 = scratch.ext(0);
                for (int i = threadIdx.x; i < N; i += blockDim.x) {
                    x0[i] = scratch.x(i);
                }
                __syncthreads();
                
            #endif

            // I.3.2. Apply min-path truncation.
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
            #if ?{traits.nodal == "block"}
                for (int i = threadIdx.x; i < N; i += blockDim.x) {
                    int i1 = i / n2;
                    int i2 = i % n2;
                    K(I1 + i1 + i2 * n2) =
                        scratch.x(i) * p_start(g1.node[i1]) * p_start(g2.node[i2]);
                }
            #elif ?{traits.nodal is True}
                #if ?{traits.diagonal is True}
                    for (int i1 = threadIdx.x; i1 < n1; i1 += blockDim.x) {
                        K(I1 + i1) = scratch.x(i1 + i1 * n1) * graphdot::ipow<2>(p_start(g1.node[i1]));
                    }
                #else
                    for (int i = threadIdx.x; i < N; i += blockDim.x) {
                        int i1 = i / n2;
                        int i2 = i % n2;
                        auto r = scratch.x(i) * p_start(g1.node[i1]) * p_start(g2.node[i2]);
                        K(I1 + i1, I2 + i2) = r;
                        #if ?{traits.symmetric is True}
                            if (job.x != job.y) K(I2 + i2, I1 + i1) = r;
                        #endif
                    }
                #endif
            #elif ?{traits.nodal is False}
                // wipe output buffer for atomic accumulations
                if (threadIdx.x == 0) {
                    #if ?{traits.diagonal is True}
                        K(I1) = 0.f;
                    #else
                        K(I1, I2) = 0.f;
                        #if ?{traits.symmetric is True}
                            K(I2, I1) = 0.f;
                        #endif
                    #endif
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
                    #if ?{traits.diagonal is True}
                        atomicAdd(K.at(I1), sum);
                    #else
                        atomicAdd(K.at(I1, I2), sum);
                        #if ?{traits.symmetric is True}
                            if (job.x != job.y) {
                                atomicAdd(K.at(I2, I1), sum);
                            }
                        #endif
                    #endif
                }
            #endif

            //------------------------------------------------------------------
            // II. Evaluate the gradient of the graph kernel with respect to
            //     hyperparameters
            //------------------------------------------------------------------
            #if ?{traits.eval_gradient is True}

                // II.1. Set up view to output buffer of kernel Jacobian tensor.
                // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                #if ?{traits.diagonal is True}
                    auto J = graphdot::tensor_view(gradient, nX, nJ);
                #else
                    auto J = graphdot::tensor_view(gradient, nX, nY, nJ);
                #endif

                // II.2. Calculate the gradients and save to the output.
                // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

                // II.2A. Nodal gradients, finite difference
                // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                #if ?{traits.nodal is True}

                    constexpr static int _offset_p = 0;
                    constexpr static int _offset_q = _offset_p + p_start.jac_dims;
                    constexpr static int _offset_v = _offset_q + 1;
                    constexpr static int _offset_e = _offset_v + node_kernel.jac_dims;

                    auto const diff = scratch.ext(1);

                    // utility

                    auto const write_jac = [&](const int k){
                        #if ?{traits.diagonal is True}
                            for(int i1 = threadIdx.x; i1 < n1; i1 += blockDim.x) {
                                J(I1 + i1, k) = diff[i1 + i1 * n1] * graphdot::ipow<2>(p_start(g1.node[i1]));
                            }
                        #else
                            for(int i = threadIdx.x; i < N; i += blockDim.x) {
                                int i1 = i / n2;
                                int i2 = i % n2;
                                const auto r = diff[i] * p_start(g1.node[i1]) * p_start(g2.node[i2]);
                                J(I1 + i1, I2 + i2, k) = r;
                                #if ?{traits.symmetric is True}
                                    if (job.x != job.y) J(I2 + i2, I1 + i1, k) = r;
                                #endif
                            }
                        #endif    
                    };

                    // dp must be done first, otherwise scratch.x will be wiped
                    // by dq, dv, and de.

                    // === dK/dp ===

                    #if ?{traits.diagonal is True}
                        for (int i1 = threadIdx.x; i1 < n1; i1 += blockDim.x) {
                            auto p1 = p_start(g1.node[i1]);
                            auto dp1 = p_start._j_a_c_o_b_i_a_n_(g1.node[i1]);
                            for(int j = 0; j < p_start.jac_dims; ++j) {
                                J(I1 + i1, _offset_p + j) = scratch.x(i1 + i1 * n1) * 2 * p1 * dp1[j];
                            }
                        }
                    #else
                        for (int i = threadIdx.x; i < N; i += blockDim.x) {
                            int i1 = i / n2;
                            int i2 = i % n2;
                            auto p1 = p_start(g1.node[i1]);
                            auto p2 = p_start(g2.node[i2]);
                            auto dp1 = p_start._j_a_c_o_b_i_a_n_(g1.node[i1]);
                            auto dp2 = p_start._j_a_c_o_b_i_a_n_(g2.node[i2]);
                            for(int j = 0; j < p_start.jac_dims; ++j) {
                                auto r = scratch.x(i) * (p1 * dp2[j] + p2 * dp1[j]);
                                J(I1 + i1, I2 + i2, _offset_p + j) = r;
                                #if ?{traits.symmetric is True}
                                    if (job.x != job.y) J(I2 + i2, I1 + i1, _offset_p + j) = r;
                                #endif
                            }
                        }
                    #endif

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
        
                    for (int i = threadIdx.x; i < N; i += blockDim.x) {
                        diff[i] = (diff[i] - scratch.x(i)) / (2 * eps_diff * q);
                    }

                    write_jac(_offset_q);
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
            
                        for (int i = threadIdx.x; i < N; i += blockDim.x) {
                            diff[i] = (diff[i] - scratch.x(i)) / (2 * eps_diff * node_kernel_flat_theta[j]);
                        }

                        write_jac(_offset_v + j);
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

                        for (int i = threadIdx.x; i < N; i += blockDim.x) {
                            diff[i] = (diff[i] - scratch.x(i)) / (2 * eps_diff * edge_kernel_flat_theta[j]);
                        }

                        write_jac(_offset_e + j);
                        __syncthreads();
                    }

                // II.2B. Graph gradients, analytic solver
                // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                #elif ?{traits.nodal is False}

                    auto jacobian = solver_t::derivative(
                        p_start,
                        node_kernel,
                        edge_kernel,
                        g1, g2,
                        scratch,
                        shmem,
                        q
                    );
                    __syncthreads();

                    #if ?{traits.diagonal is True}
                        for (int j = threadIdx.x; j < jacobian.size; j += blockDim.x) {
                            J(I1, j) = 0;
                        }
                    #else
                        for (int j = threadIdx.x; j < jacobian.size; j += blockDim.x) {
                            J(I1, I2, j) = 0;
                            #if ?{traits.symmetric is True}
                                if (job.x != job.y) J(I2, I1, j) = 0;
                            #endif
                        }
                    #endif

                    __syncthreads();

                    #if ?{traits.diagonal is True}
                        #pragma unroll (jacobian.size)
                        for(int j = 0; j < jacobian.size; ++j) {
                            auto jac = graphdot::cuda::warp_sum(jacobian[j]);
                            if (lane == 0) {
                                atomicAdd(J.at(I1, j), jac);
                            };
                        }
                    #else
                        #pragma unroll (jacobian.size)
                        for(int j = 0; j < jacobian.size; ++j) {
                            auto jac = graphdot::cuda::warp_sum(jacobian[j]);
                            if (lane == 0) {
                                atomicAdd(J.at(I1, I2, j), jac);
                                #if ?{traits.symmetric is True}
                                    if (job.x != job.y) atomicAdd(J.at(I2, I1, j), jac);
                                #endif
                            };
                        }
                    #endif

                #endif
                __syncthreads();
            #endif
        }
    }
}
