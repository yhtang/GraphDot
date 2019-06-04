import importlib
import numpy
from sklearn.gaussian_process.kernels import Kernel, Hyperparameter
from sklearn.gaussian_process.kernels import StructureOrGenericKernelMixin

try:
    from . import _kronecker_sqexp as _graphdot_impl
except:
    try:
        _graphdot_impl = importlib.import_module( '_kronecker_sqexp' )
    except:
        from graphdot import compilation_command
        import tempfile, subprocess, logging
        
        logging.info( 'Compiling GPU kernel...' )
        
        with tempfile.NamedTemporaryFile( suffix='.cu', delete=False ) as file:
            file.write(
                '''
                #include <pybind11/pybind11.h>
                #include <cuda/balloc.h>
                #include <kernel/kronecker_sqexp.h>
                
                PYBIND11_MODULE( _kronecker_sqexp, module ) {
                
                    namespace py = pybind11;
                
                    py::enum_<graphdot::cuda::belt_allocator::AllocMode>( module, "AllocMode" )
                        .value( "Device", graphdot::cuda::belt_allocator::AllocMode::Device )
                        .value( "Pinned", graphdot::cuda::belt_allocator::AllocMode::Pinned )
                        .value( "Managed", graphdot::cuda::belt_allocator::AllocMode::Managed );
                
                    py::class_<graphdot::cuda::belt_allocator>( module, "BeltAllocator" )
                        .def( py::init<std::size_t, graphdot::cuda::belt_allocator::AllocMode>(),
                              py::arg("slab_size") = 1024*1024,
                              py::arg("mode"     ) = graphdot::cuda::belt_allocator::AllocMode::Managed );
                
                    py::class_<graphdot::kernel::marginalized_kronecker_sqexp::graph_t>( module, "Graph" )
                        .def( py::init<py::tuple, graphdot::cuda::belt_allocator &, pybind11::dict>() );
                
                    py::class_<graphdot::kernel::marginalized_kronecker_sqexp>( module, "Solver" )
                        .def( py::init<py::dict>() )
                        .def( "compute", &graphdot::kernel::marginalized_kronecker_sqexp::compute );
                }
                '''.encode()    
            )
            file.flush()
            
            response = subprocess.run( compilation_command( file.name, '_kronecker_sqexp' ), shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE )
    
            if response.returncode:
                logging.error( 'GPU kernel compilation failed with error code {code}'.format( code=response.returncode ) )
    
        logging.info( 'Compilation done' )
    
        _graphdot_impl = importlib.import_module( '_kronecker_sqexp' )

class MarginalizedGraphKernel(StructureOrGenericKernelMixin, Kernel):
    
    default_runtime_config = {
        'device':       0,        # which GPU to attach to
        'block_per_sm': 8,
        'block_size':   128
    }

    def __init__(self, vertex_baseline_similarity=0.5,
                       edge_length_scale=0.1,
                       starting_probability=1.0,
                       stopping_probability=0.05,
                       vertex_baseline_similarity_bounds=(1e-7, 1),
                       edge_length_scale_bounds=(1e-7, numpy.inf),
                       starting_probability_bounds=(1e-7, numpy.inf),
                       stopping_probability_bounds=(1e-7, 1 - 1e-7),
                       **runtime_config):
 
        self.vertex_baseline_similarity        = vertex_baseline_similarity
        self.vertex_baseline_similarity_bounds = vertex_baseline_similarity_bounds

        self.edge_length_scale                 = edge_length_scale
        self.edge_length_scale_bounds          = edge_length_scale_bounds

        self.starting_probability              = starting_probability
        self.starting_probability_bounds       = starting_probability_bounds

        self.stopping_probability              = stopping_probability
        self.stopping_probability_bounds       = stopping_probability_bounds

        self.runtime_config = self.default_runtime_config
        self.runtime_config.update( **runtime_config )

        self.solver = _graphdot_impl.Solver( self.runtime_config )
 
    @property
    def hyperparameter_vertex_baseline_similarity(self):
        return Hyperparameter( "vertex_baseline_similarity", "numeric", self.vertex_baseline_similarity_bounds )

    @property
    def hyperparameter_edge_length_scale(self):
        return Hyperparameter( "edge_length_scale", "numeric", self.edge_length_scale_bounds )

    @property
    def hyperparameter_starting_probability(self):
        return Hyperparameter( "starting_probability", "numeric", self.starting_probability_bounds )

    @property
    def hyperparameter_stopping_probability(self):
        return Hyperparameter( "stopping_probability", "numeric", self.stopping_probability_bounds )

    def __call__(self, X, Y=None ):
        temporary_alloc = _graphdot_impl.BeltAllocator()
        Z1 = self._load(X, temporary_alloc)
        n1 = len( Z1 )
        if Y is None:
            i, j = numpy.triu_indices( n1 )
            dots = self.solver.compute( self.get_params(), list( zip( i, j ) ), Z1 )
            r = numpy.zeros( (n1, n1) )
            r[ i, j ] = dots
            r[ j, i ] = dots
            return r
        else:
            Z2 = self._load(Y, temporary_alloc)
            n2 = len( Z2 )
            dots = self.solver.compute( self.get_params(), [ (i, j+n1) for i in range(n1) for j in range(n2) ], Z1 + Z2 )
            return numpy.array(dots).reshape( n1, n2, order='C' )

    def diag(self, X):
        temporary_alloc = _graphdot_impl.BeltAllocator()
        Z = self._load(X, temporary_alloc)
        r = self.solver.compute( self.get_params(), [ (i, i) for i in range( len(Z) ) ], Z )
        return numpy.array( r )

    def _load(self, X, alloc):
        def _tranform_plain():
            pass
        def _transform_networkx():
            pass
        def _transform():
            pass

        hyperparameters = self.get_params()
        return [ x if isinstance( x, _graphdot_impl.Graph ) else _graphdot_impl.Graph( x, alloc, hyperparameters ) for x in X ]

    def is_stationary(self):
        return False

    def clone_with_theta(self):
        cloned = clone(self)
        cloned.theta = theta
        return cloned