#ifndef GRAPHDOT_GRAPH_H_
#define GRAPHDOT_GRAPH_H_

#include <cstdint>

namespace graphdot {

template<class Node, class Edge> struct graph_t {

    using real_t = float;
    using node_t = Node;
    using edge_t = Edge;
    using octile_t = struct {
        edge_t * elements;
        union {                              // column major
            std::uint64_t nzmask;            
            std::uint32_t nzmask_halves[2];  // one byte for each column
            std::uint8_t  nzmask_bytes[8];   // one byte for each column
        };
        union {                              // row major
            std::uint64_t nzmask_r;
            std::uint8_t  nzmask_r_bytes[8];
        };
        int upper, left;
    };

    constexpr static float eps = 1e-14;

    int n_node, n_octile;
    real_t * degree;
    node_t * node;
    octile_t * octile;
};

}

#endif