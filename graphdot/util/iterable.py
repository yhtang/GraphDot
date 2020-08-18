#!/usr/bin/env python
# -*- coding: utf-8 -*-


def flatten(iterable):
    '''Iterate through a tree of iterables in depth-first order. E.g.
    :py:`flatten(((1, 2), 3))` yields the sequence of :py:`1, 2, 3`.'''
    for item in iterable:
        if isinstance(item, (list, tuple)):
            yield from flatten(item)
        else:
            yield item


def fold_like(flat, example):
    '''Create a tree of iterables from an input linear sequence and a structure
    template. E.g. :py:`fold_like([1, 2, 3], ((None, None), None))` yields
    :py:`((1, 2), 3)`.'''
    folded = []
    for item in example:
        if hasattr(item, '__iter__'):
            n = len(list(flatten(item)))
            folded.append(fold_like(flat[:n], item))
            flat = flat[n:]
        else:
            folded.append(flat[0])
            flat = flat[1:]
    return tuple(folded)


def replace(iterable, old, new):
    '''Replace all occurrences of `old` to `new`.'''
    for item in iterable:
        if item == old:
            yield new
        else:
            yield item
