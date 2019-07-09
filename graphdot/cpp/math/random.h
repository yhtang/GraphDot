#ifndef GRAPHDOT_MATH_RANDOM_H_
#define GRAPHDOT_MATH_RANDOM_H_

#include <random>
#include <vector>

namespace graphdot {

namespace random {

static std::mt19937 default_rd(0);

inline void seed( std::size_t s ) {
    default_rd = std::mt19937( s );
}

template<class InputIt, class OutputIt, class RandomDevice = std::mt19937>
inline void choice( InputIt first, InputIt last, OutputIt out, std::size_t N, RandomDevice & rd = default_rd ) {

    using element_type = typename std::iterator_traits<InputIt>::value_type;

    std::vector<element_type> residual( first, last );

    N = std::min<std::size_t>( N, std::distance( first, last ) );

    for(std::size_t i = 0; i < N; ++i) {
        int p = rd() % residual.size();
        *out++ = residual[p];
        residual[p] = residual.back();
        residual.pop_back();
    }
}

//template<class OutputIt, class RandomDevice = std::mt19937>
inline double randn( double mean = 0.0, double sigma = 1.0 ) {
    return std::normal_distribution<double>( mean, sigma )( default_rd );
}

inline double randu( double vmin = 0.0, double vmax = 1.0 ) {
    return std::uniform_real_distribution<double>( vmin, vmax )( default_rd );
}

inline double lognormal( double logmean, double logstd ) {
    return std::lognormal_distribution<double>( logmean, logstd )( default_rd );
}

inline double loguniform( double vmin, double vmax ) {
    return std::exp( std::uniform_real_distribution<double>( 0.0, std::log(vmax) - std::log(vmin) )( default_rd ) + std::log(vmin) );
}

}

}

#endif
