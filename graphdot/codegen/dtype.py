from graphdot.codegen import Template
import numpy


def as_struct(type):
    type = numpy.dtype(type, align=True)  # convert numpy.float32 etc. to dtype
    if type.names is not None:
        return Template('''struct { ${members;}; }''').render(
            members=['{} {}'.format(as_struct(t), v)
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
        print(as_struct(type))
