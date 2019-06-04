#ifndef GRAPHDOT_CPP_SPATIAL_RCM_ORDER_H_
#define GRAPHDOT_CPP_SPATIAL_RCM_ORDER_H_

#include <algorithm>
#include <iostream>
#include <vector>
#include <Eigen/Dense>

#include <ext/json/json.hpp>

namespace graphdot {

namespace spatial {

// @OGUZ-TODO ensure symmetricity
std::vector<int32_t> rcm_order ( json::json const & graph ) {

    std::size_t nverts = graph["vertices"].size();

    std::vector<double> adj( nverts * nverts );

    for ( auto const &e: graph["edges"] ) {
        std::size_t i = e["i"];
        std::size_t j = e["j"];
        float w = e["w"];
        adj[ i * nverts + j ] = w;
        adj[ j * nverts + i ] = w;
    }

    int32_t         *        deg     = new int32_t[nverts]();
    int32_t         *        Q       = new int32_t[nverts]();
    bool            *        visited = new bool[nverts]();
    std::vector<int32_t>     perm( nverts );

    for ( std::size_t v = 0; v < nverts; ++v ) {
        for ( std::size_t u = 0; u < nverts; ++u ) {
            if ( adj[u * nverts + v] != ( double )0.0 )
                ++( deg[v] );
        }
    }

    int32_t qbeg = 0, qend = 0;
    int32_t pidx = nverts - 1;

    while ( 1 ) {
        int32_t mindeg = std::numeric_limits<int32_t>::max();
        int32_t v      = -1;
        for ( std::size_t u = 0; u < nverts; ++u ) {
            if ( !visited[u] && deg[u] < mindeg ) {
                mindeg = deg[u];
                v      = u;
            }
        }

        if ( v == -1 )
            break;

        Q[qend]    = v;
        visited[v] = true;
        ++qend;
        while ( qbeg < qend ) {
            v = Q[qbeg];
            ++qbeg;

            std::vector<std::pair<int32_t, int32_t> > adjx;
            for ( std::size_t u = 0; u < nverts; ++u ) {
                if ( !visited[u] && u != v && adj[u * nverts + v] != ( double )0.0 )
                    adjx.push_back( std::make_pair( deg[u], u ) );
            }
            std::sort( adjx.begin(), adjx.end() );

            for ( auto const &pair : adjx ) {
                auto u = pair.second;
                Q[qend++]  = u;
                visited[u] = true;
            }

            perm[pidx--] = v;
        }
    }

    delete [] deg;
    delete [] Q;
    delete [] visited;

    return perm;
}

}

}

#endif
