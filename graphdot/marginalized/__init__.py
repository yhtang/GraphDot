import sys
sys.path.append('/home/ytang/Seafile/research/source/graphdot')


import numpy
import pycuda
import pycuda.gpuarray
import pycuda.autoinit
from graphdot.codegen import Template


supported_basetypes = {
    int: 'int',
    float: 'float',
    bool: 'bool',
    numpy.bool_: 'bool',
    numpy.float32: 'float',
    numpy.float64: 'double',
    numpy.int16: 'std::int16_t',
    numpy.int32: 'std::int32_t',
    numpy.int64: 'std::int64_t',
    numpy.dtype(numpy.bool_): 'bool',
    numpy.dtype(numpy.float32): 'float',
    numpy.dtype(numpy.float64): 'double',
    numpy.dtype(numpy.int16): 'std::int16_t',
    numpy.dtype(numpy.int32): 'std::int32_t',
    numpy.dtype(numpy.int64): 'std::int64_t',
}

def get_accessor(dtype, parent):
    if isinstance(dtype, numpy.dtype):
        if dtype.fields is not None:
            return {key: get_accessor(dtype.fields[key][0],
                                      'get<%d>(%s)' % (i, parent))
                    for i, key in enumerate(dtype.fields)}
        else:
            return parent
    elif isinstance(dtype, (list, tuple)):
        return [get_accessor(d, 'get<%d>(%s)' % (i, parent))
                for i, d in enumerate(dtype)]
    elif dtype in supported_basetypes:
        return parent
    else:
        raise TypeError('type ' + repr(dtype) + ' is not allowed.')


def gencode_kvert(dtype):
    def gencode(vertex, kernel):
        return kernel.gencode(get_accessor(vertex, 'X'),
                              get_accessor(vertex, 'Y'))


def compute(graph, kernel):
    """
    todo:
    for [nodes, edges]:
        1. infer attribute structure and layout
        2. pack_into attributes
        3. kernel code gen based on layout of attributes
    """
    print('===============================================')

    df = graph.nodes

    packing = numpy.argsort([df.dtypes[key].itemsize for key in df.columns])
    packed_attributes = [df.columns[i] for i in packing[-1::-1]]
    packed_dtype = numpy.dtype([(key, df.dtypes[key].newbyteorder('='))
                                for key in packed_attributes], align=True)
    print('dtype\n', packed_dtype)

    print(Template(r'''
    using vert_t = ${vert_t};
    ''').render(vert_t=decltype(packed_dtype)))

    print(Template(r'''
    struct vertex_kernel {
        template<class V>
        static auto compute(V const &v1, V const &v2) {
            ${statement};
        }
    };''').render(statement=kernel.gencode(get_accessor(packed_dtype, 'v1'),
                                           get_accessor(packed_dtype, 'v2'))))

    node_gpu = pycuda.gpuarray.GPUArray(df.shape[0], packed_dtype)
    print(repr(node_gpu))


if __name__ == '__main__':

    if True:
        import networkx as nx

        # from graphdot.marginalized.basekernel import Constant
        from graphdot.marginalized.basekernel import KroneckerDelta
        from graphdot.marginalized.basekernel import SquareExponential
        from graphdot.marginalized.basekernel import KeywordTensorProduct
        # from graphdot.marginalized.basekernel import Convolution

        from graphdot import Graph

        class Hybrid:
            NONE = 0
            SP = 1
            SP2 = 2
            SP3 = 3

        g = nx.Graph(title='H2O')
        g.add_node('O1', hybridization=Hybrid.SP2, charge=1, conjugate=False)
        g.add_node('H1', hybridization=Hybrid.SP3, charge=-1, conjugate=True)
        g.add_node('H2', hybridization=Hybrid.SP, charge=2, conjugate=True)
        # g.add_node('H2', hybridization=Hybrid.SP, charge=2, time=1)
        g.add_edge('O1', 'H1', order=1, length=0.5)
        g.add_edge('O1', 'H2', order=2, length=1.0)

        gg = Graph.from_networkx(g)

        kv = KeywordTensorProduct(hybridization=KroneckerDelta(0.3, 1.0),
                                  charge=SquareExponential(1.0),
                                  conjugate=KroneckerDelta(0.5))

        print(gg)

        compute(gg, kv)
