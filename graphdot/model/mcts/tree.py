#!/usr/bin/env python
# -*- coding: utf-8 -*-
from graphdot.minipandas import DataFrame


class Tree(DataFrame):

    class NodeView:
        def __init__(self, tree, i):
            self._tree = tree
            self._i = i

        def __getattr__(self, key):
            if key.startswith('_'):
                return self.__dict__[key]
            else:
                return self._tree[key][self._i]

        def __setattr__(self, key, val):
            if key.startswith('_'):
                self.__dict__[key] = val
            else:
                self._tree[key][self._i] = val

        def __str__(self):
            return ' '.join([f'{a}:{getattr(self, a)}' for a in self._tree.columns])

    def __init__(self, d={}, **kwargs):
        d.update(**kwargs)
        super().__init__(d)

    def iternodes(self):
        for i in range(len(self)):
            yield Tree.NodeView(self, i)

    def __str__(self):
        return '\n'.join([
            # str(t) if c is None else f'{t}\n\tXXX' for t, c in zip(
                str(t) if c is None else f'{t}\n\t' + '\n\t'.join(str(c).split('\n')) for t, c in zip(
            # f'{t}' for t, c in zip(
                self.drop(['parent', 'children']).iternodes(),
                self.children
            )
        ])

        
