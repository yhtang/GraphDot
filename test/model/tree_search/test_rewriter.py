#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest
import numpy as np
from graphdot.model.tree_search import LookAheadSequenceRewriter


def test_sequence_rewriter_init():
    rw = LookAheadSequenceRewriter(random_state=0)
    rw = LookAheadSequenceRewriter(random_state=np.random.default_rng())

    with pytest.raises(RuntimeError):
        rw.tree.show()
    rw.fit(['A'])
    rw.tree.show()


def test_sequence_rewriter_fit():
    rw = LookAheadSequenceRewriter(n=0)
    rw.fit(['A'])
    assert len(rw.tree.nodes) == 2
    a, = rw.tree.children(rw.tree.root)
    assert a.data.count == 1
    assert a.data.freq == 1.0

    rw = LookAheadSequenceRewriter(n=0)
    rw.fit(['A', 'B'])
    assert len(rw.tree.nodes) == 3
    a, b = rw.tree.children(rw.tree.root)
    assert a.data.count == 1
    assert a.data.freq == 0.5
    assert b.data.count == 1
    assert b.data.freq == 0.5

    rw = LookAheadSequenceRewriter(n=0)
    rw.fit(['AABBB'])
    assert len(rw.tree.nodes) == 3
    a, b = rw.tree.children(rw.tree.root)
    assert a.data.count == 2
    assert a.data.freq == 0.4
    assert b.data.count == 3
    assert b.data.freq == 0.6

    rw = LookAheadSequenceRewriter(n=1)
    rw.fit(['A', 'B'])
    assert len(rw.tree.nodes) == 3
    a, b = rw.tree.children(rw.tree.root)
    assert a.data.count == 1
    assert a.data.freq == 0.5
    assert b.data.count == 1
    assert b.data.freq == 0.5

    rw = LookAheadSequenceRewriter(n=1)
    rw.fit(['AA', 'BB'])
    assert len(rw.tree.nodes) == 5
    a, b = rw.tree.children(rw.tree.root)
    aa, = rw.tree.children(a.identifier)
    bb, = rw.tree.children(b.identifier)
    assert a.data.count == 2
    assert a.data.freq == 0.5
    assert b.data.count == 2
    assert b.data.freq == 0.5
    assert aa.data.count == 1
    assert aa.data.freq == 1.0
    assert bb.data.count == 1
    assert bb.data.freq == 1.0

    rw = LookAheadSequenceRewriter(n=0)
    rw.fit(['ABCDE'])
    assert len(rw.tree.nodes) == 6
    for n in rw.tree.children(rw.tree.root):
        assert n.data.count == 1
        assert n.data.freq == 0.2

    rw = LookAheadSequenceRewriter(n=1)
    rw.fit(['ABCDE'])
    assert len(rw.tree.nodes) == 10
    for n in rw.tree.children(rw.tree.root):
        assert n.data.count == 1
        assert n.data.freq == 0.2
        if n.tag == 'E':
            assert rw.tree.children(n.identifier) == []
        else:
            nn, = rw.tree.children(n.identifier)
            assert nn.data.count == 1
            assert nn.data.freq == 1.0


def test_sequence_rewriter_context_match():
    rw = LookAheadSequenceRewriter(n=3)
    rw.fit(['ABCDEF'])
    assert rw._match_context(rw.tree, 'ABCDEF', 0, rw.n).tag == '$'
    assert rw._match_context(rw.tree, 'ABCDEF', 1, rw.n).tag == 'A'
    assert rw._match_context(rw.tree, 'ABCDEF', 2, rw.n).tag == 'B'
    assert rw._match_context(rw.tree, 'ABCDEF', 3, rw.n).tag == 'C'

    rw = LookAheadSequenceRewriter(n=3)
    rw.fit(['AXABXABCXABCDX'])
    assert rw.tree.parent(
        rw._match_context(rw.tree, 'AX', 2, 3).identifier
    ).tag == 'A'
    assert rw.tree.parent(
        rw._match_context(rw.tree, 'ABX', 3, 3).identifier
    ).tag == 'B'
    assert rw.tree.parent(
        rw._match_context(rw.tree, 'ABCX', 4, 3).identifier
    ).tag == 'C'
    # due to lack of appending symbols
    assert rw._match_context(rw.tree, 'ABCDX', 5, 3) is None


def test_sequence_rewriter_insert():
    rw = LookAheadSequenceRewriter(n=0)
    rw.fit(['A'])
    s = 'BBB'
    assert rw._insert(s, 0) == 'ABBB'
    assert rw._insert(s, 1) == 'BABB'
    assert rw._insert(s, 2) == 'BBAB'
    assert rw._insert(s, 3) == 'BBBA'

    rw = LookAheadSequenceRewriter(n=0)
    rw.fit(['A', 'B'])
    s = 'CCC'
    T = [rw._insert(s, 0) for _ in range(10000)]
    fA = np.count_nonzero([t.startswith('A') for t in T]) / len(T)
    fB = np.count_nonzero([t.startswith('B') for t in T]) / len(T)
    assert fA + fB == pytest.approx(1.0)
    assert fA >= .475 and fA <= .525
    assert fB >= .475 and fB <= .525

    rw = LookAheadSequenceRewriter(n=2)
    rw.fit(['AAB', 'BAA'])
    for i in range(100):
        assert rw._insert('AA', 1) in ['AAA', 'ABA']
    assert rw._insert('AA', 2) == 'AAB'
    assert rw._insert('BA', 1) == 'BAA'
    assert rw._insert('BA', 2) == 'BAA'


def test_sequence_rewriter_mutate():
    rw = LookAheadSequenceRewriter(n=0)
    rw.fit(['A'])
    s = 'BBB'
    assert rw._mutate(s, 0) == 'ABB'
    assert rw._mutate(s, 1) == 'BAB'
    assert rw._mutate(s, 2) == 'BBA'

    rw = LookAheadSequenceRewriter(n=0)
    rw.fit(['A', 'B'])
    s = 'CCC'
    T = [rw._mutate(s, 0) for _ in range(10000)]
    fA = np.count_nonzero([t.startswith('A') for t in T]) / len(T)
    fB = np.count_nonzero([t.startswith('B') for t in T]) / len(T)
    assert fA + fB == pytest.approx(1.0)
    assert fA >= .475 and fA <= .525
    assert fB >= .475 and fB <= .525

    rw = LookAheadSequenceRewriter(n=2)
    rw.fit(['AAB', 'BA'])
    for _ in range(100):
        assert rw._mutate('AAC', 1) in ['ABC', 'AAC']
    assert rw._mutate('AAC', 2) == 'AAB'
    assert rw._mutate('BAB', 1) == 'BAB'
    for _ in range(100):
        assert rw._mutate('BAB', 2) in ['BAA', 'BAB']


def test_sequence_rewriter_delete():
    rw = LookAheadSequenceRewriter(n=0)
    rw.fit(['A'])
    s = 'ABCDE'
    assert rw._delete(s, 0) == 'BCDE'
    assert rw._delete(s, 1) == 'ACDE'
    assert rw._delete(s, 2) == 'ABDE'
    assert rw._delete(s, 3) == 'ABCE'
    assert rw._delete(s, 4) == 'ABCD'


def test_sequence_rewriter_call():
    for b in [1, 2, 3, 5, 10]:
        rw = LookAheadSequenceRewriter(n=0, b=b)
        rw.fit(['AA'])
        s = 'ABC'
        T = rw(s)
        assert len(T) <= b
        for t in T:
            assert t != s
