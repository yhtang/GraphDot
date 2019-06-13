from graphdot.interop.structure import flatten

import itertools
import struct
import pytest


nested_layout_examples = [
    ('[[b][b]]', 'bb'),
    ('[[b][h]]', 'bxh'),
    ('[[b][f]]', 'bxxxf'),
    ('[[b][d]]', 'bxxxxxxxd'),
    ('[[bh][hdf]]', 'bxhxxxxhxxxxxxdfxxxx'),
    ('[[ff][f]]', 'fff'),
]


def test_flatten_shallow():
    ''' all combinations of four member structus '''
    vals = (1, 2, 3, 4)
    for perm in itertools.product('bhfd', repeat=4):
        layout = ''.join(perm)
        flat, alignment = flatten('[{}]'.format(layout))
        if 'd' in layout:
            layout_trail0 = layout + '0d'
        elif 'f' in layout:
            layout_trail0 = layout + '0f'
        elif 'h' in layout:
            layout_trail0 = layout + '0h'
        else:
            layout_trail0 = layout
        assert(struct.calcsize(flat) == struct.calcsize(layout_trail0))
        assert(struct.unpack(layout_trail0, struct.pack(flat, *vals)) == vals)

    ''' random combinations '''
    for n in range(6, 20):
        vals = tuple(range(n))
        for comb in itertools.combinations_with_replacement('bhfd', n):
            print(comb)
            layout = ''.join(comb)
            flat, alignment = flatten('[{}]'.format(layout))
            assert(struct.unpack(layout, struct.pack(flat, *vals)) == vals)


@pytest.mark.parametrize('example', nested_layout_examples)
def test_flatten_nested(example):
    assert(flatten(example[0])[0] == example[1])
