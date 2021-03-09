#!/usr/bin/env python
# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
import itertools as it
from collections import deque
import numpy as np
from treelib import Tree


class AbstractRewriter(ABC):
    ''' Abstract base class for graph rewrite rules. '''

    @abstractmethod
    def __call__(self, g):
        ''' Rewrite the given graph using a rule drawn randomly from a pool.

        Parameters
        ----------
        g: object
            An input graph to be transformed.

        Returns
        -------
        H: list
            A list of new graphs transformed from `g`.
        '''


class LookAheadSequenceRewriter(AbstractRewriter):
    '''A sequence rewriter that performs contextual updates to a symbol
    sequence using the n-gram preceding the location of modification. It can
    carry out three types of operations:

    - Insertion: insert an symbol at a random location. The symbol inserted
      should be probabilistically determined by up to **n** items in front of
      it unless when there are less than n symbols in the front, or when there
      is no matching n-gram in the training set. In that case, the longest
      matching k-gram (k < n) is used.

    - Mutation: replace an symbol by a random one. This is context-sensitive.

    - Deletion: remove an symbol at random from a sequence.
      This is context-insensitive.

    Parameters
    ----------
    n: int
        The maximum number of items to look ahead for contextual rewrites.
    b: int
        The branching factor, i.e. the number of new sequences to create from
        each input sequence.
    min_edits: int
        The minimum number of edits made to create a new sequence.
    max_edits: int
        The maximum number of edits made to create a new sequence.
    p: list of three numbers
        The relative frequencies of insertation, mutation, and deletion
        operations.
    random_state: np.random.Generator or int
        Initial state for the internal RNG.
    '''

    class Payload:
        def __init__(self, **kwargs):
            self.__dict__.update(**kwargs)

    def __init__(self, n=1, b=3, min_edits=1, max_edits=5, p_insert=1,
                 p_mutate=1, p_delete=1, random_state=None):
        self.n = n
        self.b = b
        self.p_imd = np.array([p_insert, p_mutate, p_delete], dtype=np.float)
        self.p_imd /= self.p_imd.sum()
        self.min_edits = min_edits
        self.max_edits = max_edits
        self.rng = self._parse_random_state(random_state)

    @staticmethod
    def _parse_random_state(random_state):
        if isinstance(random_state, np.random.Generator):
            return random_state
        elif random_state is not None:
            return np.random.Generator(np.random.PCG64(random_state))
        else:
            return np.random.default_rng()

    @property
    def tree(self):
        '''A tree-representation of the 1- to n-gram distributions of the
        training set.'''
        try:
            return self._tree
        except AttributeError:
            raise RuntimeError(
                'The rewriter must be trained on a collection of sequences '
                'first using the ``fit()`` method.'
            )

    @tree.setter
    def tree(self, tree):
        self._tree = self._recursive_normalize(Tree(tree, deep=True))

    def _recursive_normalize(self, tree, nid=None):
        nid = nid or tree.root
        children = tree.children(nid)
        counts = np.array([c.data.count for c in children])
        freqs = counts / np.sum(counts)
        for c, f in zip(children, freqs):
            c.data.freq = f
        for c in children:
            self._recursive_normalize(tree, c.identifier)
        return tree

    def fit(self, X):
        '''Learn the n-gram distribution from the given dataset.

        Parameters
        ----------
        X: list of sequences
            The training set.
        '''
        tree = Tree()
        root = tree.create_node('$', data=self.Payload(count=0, freq=0))
        for seq in X:
            ptrs = deque()
            for symbol in seq:
                ptrs.append(root)
                if len(ptrs) > self.n + 1:
                    ptrs.popleft()
                for i, p in enumerate(ptrs):
                    try:
                        next, = [c for c in tree.children(p.identifier)
                                 if c.tag == symbol]
                        next.data.count += 1
                    except ValueError:
                        next = tree.create_node(
                            tag=symbol, parent=p.identifier,
                            data=self.Payload(count=1, freq=0)
                        )
                    ptrs[i] = next
        self.tree = tree

    @staticmethod
    def _match_context(tree, s, k, n):
        ptrs = [tree[tree.root] for _ in range(n + 1)]
        for i, loc in enumerate(range(max(k - n, 0), k)):
            for j, p in enumerate(ptrs[:i + 1]):
                if p is not None:
                    try:
                        next, = [c for c in tree.children(p.identifier)
                                 if c.tag == s[loc]]
                    except (KeyError, ValueError):
                        next = None
                    ptrs[j] = next
        for n in ptrs:
            if n is not None and len(tree.children(n.identifier)) > 0:
                return n

    def _propose(self, s, k):
        cxt = self._match_context(self.tree, s, k, self.n)
        children = self.tree.children(cxt.identifier)
        freq = np.array([c.data.freq for c in children])
        return self.rng.choice(children, p=freq).tag

    def _insert(self, s, k):
        return s[:k] + type(s)(self._propose(s, k)) + s[k:]

    def _mutate(self, s, k):
        return s[:k] + type(s)(self._propose(s, k)) + s[k + 1:]

    def _delete(self, s, k):
        return s[:k] + s[k + 1:]

    def _rewrite(self, s):
        '''Rewrite a sequence once by randomly choosing between insertion,
        deletion, and mutation actions.

        Parameters
        ----------
        s: sequence
            The sequence to be rewritten.

        Returns
        -------
        t: sequence
            An offspring sequence
        '''
        op = self.rng.choice(
            [self._insert, self._mutate, self._delete], p=self.p_imd
        )
        k = self.rng.choice(len(s))
        return op(s, k)

    def __call__(self, s):
        '''Generate ``b`` offspring sequences, each being rewritten at least
        ``min_edits`` and at most ``max_edits`` times.

        Parameters
        ----------
        s: sequence
            The sequence to be rewritten.

        Returns
        -------
        T: list of sequences
            A collection of unique offspring sequences
        '''
        offspring = set([s])
        for t in it.repeat(s, self.b):
            for i in range(self.max_edits):
                t = self._rewrite(t)
                if i >= self.min_edits - 1 and t not in offspring:
                    offspring.add(t)
                    break
        offspring.remove(s)
        return list(offspring)
