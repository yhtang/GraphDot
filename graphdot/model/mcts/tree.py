#!/usr/bin/env python
# -*- coding: utf-8 -*-
from graphdot.minipandas import DataFrame


class Tree(DataFrame):

    class NodeView:
        def __init__(self, tree, i):
            self.__dict__.update(tree=tree, i=i)

        def __getattr__(self, key):
            return self.tree[key][self.i]

        def __setattr__(self, key, val):
            self.tree[key][self.i] = val

        def __str__(self):
            return ' '.join([
                f'{key}:{getattr(self, key)}' for key in self.tree.columns
            ])

    def __init__(self, data={}, **kwargs):
        data.update(**kwargs)
        super().__init__(data)

    def iternodes(self):
        for i in range(len(self)):
            yield Tree.NodeView(self, i)

    def _flatten(self, payloads, level=0):
        entries = []
        for children, entry in zip(
            self.children, zip(*[self[key] for key in payloads])
        ):
            entries.append([level, *entry])
            if children is not None:
                entries += children._flatten(payloads, level=level + 1)
        return entries

    @property
    def flat(self):
        payloads = [a for a in self.columns if a not in ['parent', 'children']]
        return DataFrame(
            {key: val for key, val in zip(
                ['level'] + payloads,
                list(zip(*self._flatten(payloads)))
            )}
        )

    def __str__(self):
        return '\n'.join([
            '  ' * n.level + str(n) for n in self.flat.itertuples('node')
        ])
