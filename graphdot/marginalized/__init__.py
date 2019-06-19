# from graphdot.marginalized.basekernel import Constant
from graphdot.marginalized.basekernel import KroneckerDelta
from graphdot.marginalized.basekernel import SquareExponential
from graphdot.marginalized.basekernel import TensorProduct
# from graphdot.marginalized.basekernel import Convolution

"""
['C', 2.0] => x[0], x[1]

"""


def get_format(object):
    if isinstance(object, list) or isinstance(object, tuple):
        return '[' + ''.join([get_format(element) for element in object]) + ']'
    elif isinstance(object, int):
        return 'i'
    elif isinstance(object, float):
        return 'f'
    elif isinstance(object, bool):
        return '?'
    else:
        raise TypeError(repr(object) + ' of invalid type for vertex label')


def get_accessor(object, base):
    if isinstance(object, (list, tuple)):
        return [get_accessor(element, 'get<%d>(%s)' % (i, base))
                for i, element in enumerate(object)]
    elif isinstance(object, (int, float, bool)):
        return base
    else:
        raise TypeError(repr(object) + ' of invalid type for vertex label')


def gencode(vertex, kernel):
    return kernel.gencode(get_accessor(vertex, 'X'),
                          get_accessor(vertex, 'Y'))


label_vertex = [True, 2.0, True]
kernel_vertex = TensorProduct(KroneckerDelta(0.3, 1.0),
                              SquareExponential(1.0),
                              KroneckerDelta(0.7, 1.0))

print(get_format(label_vertex))
print(get_accessor(label_vertex, 'X'))
print(gencode(label_vertex, kernel_vertex))
