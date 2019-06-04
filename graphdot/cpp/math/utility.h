#ifndef GRAPHDOT_CPP_MISC_UTILITY_H
#define GRAPHDOT_CPP_MISC_UTILITY_H

namespace graphdot {

template<class T> constexpr inline T ipow ( T x, unsigned int y ) noexcept {
	return y ? x * ipow( x, y-1 ) : 1;
}

}

#endif
