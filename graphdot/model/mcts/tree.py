#!/usr/bin/env python
# -*- coding: utf-8 -*-
import itertools as it
import numpy as np
from graphdot.codegen.template import Template
from graphdot.minipandas import DataFrame


class TreeDF(DataFrame):

    class NodeProxy:
        ''' A MCTS tree node.

        Parameters
        ----------
        parent: TreeNode
            The current Node's parent. None for the root node.
        children: list of TreeNode
            All the child nodes of the current node.
        '''

        def __init__(self, parent=None, children=None):
            self.parent = parent
            self.children = children

        @property
        def children(self):
            return self._children

        @children.setter
        def children(self, iterable=None):
            if iterable is None:
                self._children = None
            else:
                self._children = DataFrame(dict(
                    nodes=list(iterable)
                ))

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    

    # def __repr__(self):
    #     if self.children:
    #         return Template(r'${node}\n${children\n\t}').render(
    #             node=self.payload,
    #             children=it.chain.from_iterable([
    #                 repr(c).split('\n') for c in self.children
    #             ])
    #         )
    #     else:
    #         return Template(r'${node}').render(
    #             node=self.payload,
    #         )


