import os, sysconfig
import cppyy

rc = dict()
rc['nvcc'] = 'nvcc'
rc['nvcc_option'] = ['-std=c++14',
                      '-O4',
                      '--expt-relaxed-constexpr',
                      '--use_fast_math',
                      '--maxrregcount 64',
                      '-Xptxas -v',
                      '-Xcompiler -fPIC',
                      '-shared' ]
rc['lib'    ] = [ '-lcudart', '-lrt' ]
rc['arch'   ] = [ 'compute_30', 'compute_50', 'compute_60', 'compute_70' ]
rc['code'   ] = [ 'sm_30', 'sm_50', 'sm_60', 'sm_70' ]
rc['include'] = [ sysconfig.get_path('include'),
                  sysconfig.get_path('platinclude'),
                  os.path.join( os.path.dirname( os.path.abspath( __file__ ) ), 'cpp' ) ]  
rc['suffix' ] = sysconfig.get_config_var('EXT_SUFFIX')
rc['cpp'    ] = os.path.join( os.path.dirname( os.path.abspath( __file__ ) ), 'cpp' )  

try:
    import pybind11
    rc['include'].append( pybind11.get_include() )
    rc['include'].append( pybind11.get_include(True) )
except:
    import logging
    logging.warning( 'pybind11 not available, runtime compilation may fail' )


def compilation_command( input, output ):
    line = '{nvcc} {include} {option} {input} {lib} {gencode} -o {output}'.format(
        nvcc    = rc['nvcc'],
        include = ' '.join( [ '-I%s' % dir for dir in rc['include'] ] ),
        option  = ' '.join( rc['nvcc_option'] ),
        input   = input,
        lib     = ' '.join( rc['lib'] ),
        gencode = ' '.join( [ '--generate-code arch=%s,code=%s' % (arch,code) for arch, code in zip( rc['arch'], rc['code'] ) ] ),
        output  = output+rc['suffix']
    )
    return line

cppyy.add_include_path( rc['cpp'] )
cppyy.include( os.path.join( rc['cpp'], 'cuda/balloc.h' ) )
cppyy.include( os.path.join( rc['cpp'], 'kernel/elementary.h' ) )

BeltAllocator = cppyy.gbl.graphdot.cuda.belt_allocator