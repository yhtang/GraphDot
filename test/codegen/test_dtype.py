from graphdot.codegen.typetool import decltype
import numpy
import pytest

composite1 = numpy.dtype([('field1', numpy.float32), ('field2', numpy.int16)])
composite2 = numpy.dtype([('over1', composite1), ('over2', numpy.bool_)])
dtypes = [
    int, float, bool,
    numpy.bool_,
    numpy.int32,
    numpy.float64,
    composite1,
    composite2
]


# @pytest.mark.parametrize('dtype', dtypes)
# def test_accessor_generator(dtype):
#     print(get_accessor(dtype, 'X'))


# @pytest.mark.parametrize('dtype', [numpy.float64])
@pytest.mark.parametrize('dtype', dtypes)
def test_decltype(dtype):
    print(decltype(dtype))


# types = [
#     numpy.dtype(numpy.float32),
#     numpy.dtype([('a', numpy.float32)]),
#     numpy.dtype([('a', numpy.dtype([('X', numpy.float32), ('Y', 'i8')])),
#                  ('b', numpy.bool_)]),
# ]
#
# for type in types:
#     print(decltype(type))
