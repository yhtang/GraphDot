#ifndef GRAPHDOT_NUMPY_TYPE_H_
#define GRAPHDOT_NUMPY_TYPE_H_

#include <cstdint>

namespace graphdot {

namespace numpy_type {
    using _empty = struct {};
    using bool_ = bool;
    using int_ = long;
    using intc = int;
    using intp = std::size_t;
    using uint8 = std::uint8_t;
    using uint16 = std::uint16_t;
    using uint32 = std::uint32_t;
    using uint64 = std::uint64_t;
    using int8 = std::int8_t;
    using int16 = std::int16_t;
    using int32 = std::int32_t;
    using int64 = std::int64_t;
    using float_ = double;
    using float32 = float;
    using float64 = double;
}

}

#endif
