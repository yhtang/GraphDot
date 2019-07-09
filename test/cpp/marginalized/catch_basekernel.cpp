#include <catch.hpp>
#include <iostream>
#include <string>

#define __GENERIC__ inline

#include <marginalized/basekernel.h>
#include <misc/metadbg.h>

TEST_CASE( "Tag creation", "[tag][creation]" ) {

    using namespace graphdot::basekernel;

    auto k1 = constant{1.0};
    auto k2 = kronecker_delta{0.5, 1.0};
    auto k = tensor_product<std::tuple<decltype(k1), decltype(k2)>, 0, 1>{ std::make_tuple(k1, k2) };
    auto g = convolution<decltype(k2)>{ k2 };

    std::cout << k( std::make_tuple('a', 1), std::make_tuple('b', 2) ) << std::endl;
    std::cout << g( std::array<int,2>{1, 2}, std::array<int,3>{2, 3, 4} ) << std::endl;
}
