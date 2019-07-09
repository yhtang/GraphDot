#ifndef GRAPHDOT_MISC_FORMAT_H_
#define GRAPHDOT_MISC_FORMAT_H_

#include <string>

namespace graphdot {

template<typename ...T> inline std::string format( std::string fmt, T... args ) {
    char buf[10240];
    snprintf ( buf, 10240, fmt.c_str(), args... );
    return std::string ( buf );
}

}

#endif
