from graphdot.codegen import Template
import numpy


def decltype(type):
    type = numpy.dtype(type, align=True)  # convert numpy.float32 etc. to dtype
    if type.names is not None:
        return Template('''struct{${members;};}''').render(
            members=['{} {}'.format(decltype(t), v)
                     for v, (t, offset) in type.fields.items()])
    else:
        return str(type.name)


if __name__ == '__main__':

    types = [
        numpy.dtype(numpy.float32),
        numpy.dtype([('a', numpy.float32)]),
        numpy.dtype([('a', numpy.dtype([('X', numpy.float32), ('Y', 'i8')])),
                     ('b', numpy.bool_)]),
    ]

    for type in types:
        print(decltype(type))
