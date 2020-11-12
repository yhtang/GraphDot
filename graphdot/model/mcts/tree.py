#!/usr/bin/env python
# -*- coding: utf-8 -*-
import itertools as it
from graphdot.codegen.template import Template


class TreeNode:
    ''' A MCTS tree node.

    Parameters
    ----------
    parent: TreeNode
        The current Node's parent. None for the root node.
    children: list of TreeNode
        All the child nodes of the current node.
    payload: kwargs
        Additional information to be carried by the node.
    '''

    def __init__(self, parent=None, children=None, **payload):
        self.parent = parent
        self.children = children
        self.payload = dict(**payload)

    def __getattr__(self, key):
        if key in self.payload:
            return self.payload[key]
        else:
            raise AttributeError(f'Cannot find attribute {key}')

    def __repr__(self):
        if self.children:
            return Template(r'${node}\n${children\n\t}').render(
                node=self.payload,
                children=it.chain.from_iterable([
                    repr(c).split('\n') for c in self.children
                ])
            )
        else:
            return Template(r'${node}').render(
                node=self.payload,
            )
