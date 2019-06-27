#ifndef GRAPHDOT_MISC_TIMER_H_
#define GRAPHDOT_MISC_TIMER_H_

#include <iostream>
#include <iomanip>
#include <string>
#include <map>
#include <unordered_map>
#include <chrono>

namespace graphdot {

inline void display_timing( std::ostream & out, double reading, const char msg[], bool human = true ) {
    std::ios::fmtflags f( out.flags() );

    std::string unit = "  s ";
    if ( human ) {
        if ( reading < 9e-4 ) {
            reading *= 1e6;
            unit = " us ";
        } else if ( reading < 9e-1 ) {
            reading *= 1e3;
            unit = " ms ";
        }
    }

    out << std::fixed << std::setprecision( 7 ) << std::setw( 12 )
        << reading << unit << "on " << msg << "\n";

    out.flags( f );
}

// A single timer
struct Timer {
    using clock_t = std::chrono::high_resolution_clock;

    inline void start() const {
        _start_time = clock_t::now();
    }
    inline double stop() const {
        std::chrono::duration<double> inc = clock_t::now() - _start_time;
        _cumu_time += inc.count();
        return inc.count();
    }
    inline double read() const {
        return _cumu_time;
    }
    inline void reset() const {
        _cumu_time = 0.;
    }
private:
    mutable clock_t::time_point _start_time;
    mutable double _cumu_time = 0.0;
};

// Group of timers indexed by label
struct Timers {
    Timer    &    operator [] ( std::string const & label )       { return timers[label]; }
    Timer const & operator [] ( std::string const & label ) const { return timers[label]; }
    ~Timers() { report( true ); }
    void report( bool destructive = false ) {
        std::map<std::string, double> readings; // use map to do sort timers by label
        for ( auto & item : timers ) {
            readings[item.first] = item.second.read();
        }
        for ( auto & read : readings ) {
            display_timing( std::cout, read.second, read.first.c_str(), false );
        }
        if ( destructive ) timers.clear();
    }

protected:
    mutable std::unordered_map<std::string, Timer> timers;
};

}

#endif
