#ifndef GRAPHDOT_CUDA_UTIL_HOST_H_
#define GRAPHDOT_CUDA_UTIL_HOST_H_

#include <cstdio>
#include <csignal>
#include <string>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <misc/format.h>

namespace graphdot {

namespace cuda {

inline void verify( cudaError_t err ) {
    if ( err ) {
        fprintf( stderr, "%s\n", cudaGetErrorString( err ) );
        raise( SIGABRT );
    }
}

inline void sync_and_peek( const char * cmsg, const int imsg ) {
#ifndef NDEBUG
    cudaDeviceSynchronize();
    fprintf( stderr, "%s %s %d\n", cudaGetErrorString( cudaGetLastError() ), cmsg, imsg );
#endif
}

inline int detect_cuda( int device = 0 )
{
    auto smver2core = [](int major, int minor){
        switch(major){
            case 1:  return 8;
            case 2:  switch(minor){
                case 1:  return 48;
                default: return 32;
            }
            case 3:  return 192;
            case 6:  switch(minor){
                case 0:  return 64;
                default: return 128;
            }
            case 7: return 64;
            default: return -1;
        }
    };

    // check capabilities
    cudaDeviceProp prop;
    verify( cudaGetDeviceProperties( &prop, device ) );
    // display information
    std::string info;
    info += format( "Device %d: %s", device, prop.name );
    info += ", ";
    info += format( "Compute Capability %d.%d", prop.major, prop.minor );
    info += "\n";
    info += format( "%.2f GHz", prop.clockRate / 1e6 );
    info += ", ";
    info += format( "%d kB L2 cache", prop.l2CacheSize / 1024 );
    info += ", ";
    info += format( "%d MB GRAM @ %d bit | %d MHz", prop.totalGlobalMem / 1024 / 1024, prop.memoryBusWidth, prop.memoryClockRate / 1024 );
    info += "\n";
    info += format( "%d SMs, %d cores/SM", prop.multiProcessorCount, smver2core(prop.major, prop.minor) );
    info += ", ";
    info += format( "%d KB shared memory per SM", prop.sharedMemPerMultiprocessor / 1024 );
    info += ", ";
    info += format( "%d K registers per SM", prop.regsPerMultiprocessor/1024 );
    info += ", ";
    info += format( "SP/DP performance ratio %d:1", prop.singleToDoublePrecisionPerfRatio );
    info += "\n";
    info += format( "SP peak: %.2f TFLOPS", prop.multiProcessorCount * smver2core(prop.major, prop.minor) * 2.0 * prop.clockRate / 1e9  );
    info += ", ";
    info += format( "DP peak: %.2f TFLOPS", prop.multiProcessorCount * smver2core(prop.major, prop.minor) * 2.0 * prop.clockRate / 1e9 / prop.singleToDoublePrecisionPerfRatio );
    info += "\n";

    fprintf( stdout, info.c_str() );

    return 0;
}

}

}

#endif
