#include <graph.h>
#include <marginalized_kernel_cpu.h>
#include <fmath.h>
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
  void graph_kernel_solver(graph_t const   * graphs,
			   float32        ** p,
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
    uint32 i_job = -1;

    while (true) {
      i_job++;
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

      solver_t::compute(node_kernel, edge_kernel, g1, g2, scratch, q, q0);

      /********* post-processing *********/

      // apply starting probability and min-path truncation
      float iq0 = 1 / q0;
      for(int i1 = 0; i1 != n1; ++i1) {
	for(int i2 = 0; i2 != n2; ++i2) {
	  int i = n2 * i1 + i2;
	  auto r = scratch.x(i);
	  if (?{traits.lmin == 1}) {
	    r -= node_kernel(g1.node[i1], g2.node[i2]) * q * q * iq0 * iq0;
	  }
	  scratch.x(i) = r * p1[i1] * p2[i2];
	}
      }

      // write to output buffer
      if (?{traits.nodal == "block"}) {
	for(int i1 = 0; i1 != n1; ++i1) {
	  for(int i2 = 0; i2 != n2; ++i2) {	    
	    int i = n2 * i1 + i2;
	    auto r = scratch.x(i);
	    out[I1 + i1 + i2 * g1.n_node] = r;
	  }
	}
      }
      if (?{traits.nodal is True}) {
	if (?{traits.diagonal is True}) {
	  for (int i1 = 0; i1 < g1.n_node; ++i1) {
	    out[I1 + i1] = scratch.x(i1 + i1 * n1);
	  }
	} else {
	  for(int i1 = 0; i1 != n1; ++i1) {
	    for(int i2 = 0; i2 != n2; ++i2) {
	      int i = n2 * i1 + i2;
	      auto r = scratch.x(i);
	      out[(I1 + i1) + (I2 + i2) * out_h] = r;
	      if (?{traits.symmetric is True}) {
		if (job.x != job.y) out[(I2 + i2) + (I1 + i1) * out_h] = r;
	      }
	    }
	  }    
	}
      }
      
      if (?{traits.nodal is False}) {
	float32 sum = 0;
	for (int i1 = 0; i1 != n1; ++i1) {
	  for(int i2 = 0; i2 != n2; ++i2) {
	    int i = n2 * i1 + i2;
	    sum += scratch.x(i);
	  }
	}
	if (?{traits.diagonal is True}) {
	  out[I1] += sum;
	} else {
	  out[I1 + I2 * out_h] += sum;
	  if (?{traits.symmetric is True}) {
	    if (job.x != job.y) {
	      out[I2 + I1 * out_h] += sum;
	    }
	  }
	}
      }
    }
  }
}
