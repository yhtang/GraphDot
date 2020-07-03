#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
from io import StringIO
import pytest
from graphdot.util.printer import markdown


def test_markdown_header_right_justified():
    # right-align is the default
    out = sys.stdout = StringIO()
    markdown.table_header(('Hello', '%9d', 0))
    line1, line2 = out.getvalue().strip().split('\n')
    assert(line1 == '|    Hello|')
    assert(line2 == '|---------|')


def test_markdown_header_left_justified():
    # left-align
    out = sys.stdout = StringIO()
    markdown.table_header(('Hello', '%-9d', 0))
    line1, line2 = out.getvalue().strip().split('\n')
    assert(line1 == '|Hello    |')
    assert(line2 == '|---------|')


def test_markdown_header_multicol():
    out = sys.stdout = StringIO()
    markdown.table_header(
        ('Hello', '%9d', 0),
        ('Hello', '%12f', 0),
        ('Hello', '%15g', 0)
    )
    line1, line2 = out.getvalue().strip().split('\n')
    cols = line1.strip('|').split('|')
    assert(len(cols) == 3)
    assert(len(cols[0]) == 9)
    assert(len(cols[1]) == 12)
    assert(len(cols[2]) == 15)


def test_markdown_table_multicol():
    out = sys.stdout = StringIO()
    markdown.table(
        ('Hello', '%9d', 0),
        ('Hello', '%12f', 0),
        ('Hello', '%15g', 0),
        print_header=False
    )
    cols = out.getvalue().strip().strip('|').split('|')
    assert(len(cols) == 3)
    assert(len(cols[0]) == 9)
    assert(len(cols[1]) == 12)
    assert(len(cols[2]) == 15)


def test_markdown_row_print_header():
    out = sys.stdout = StringIO()
    markdown.table_start()
    markdown.table(
        ('Hello', '%9d', 0),
        ('Hello', '%12f', 0),
        ('Hello', '%15g', 0),
    )
    assert(len(out.getvalue().strip().split('\n')) == 3)

    out = sys.stdout = StringIO()
    markdown.table(
        ('Hello', '%9d', 0),
        ('Hello', '%12f', 0),
        ('Hello', '%15g', 0),
    )
    assert(len(out.getvalue().strip().split('\n')) == 1)

    out = sys.stdout = StringIO()
    markdown.table(
        ('Hello', '%9d', 0),
        ('Hello', '%12f', 0),
        ('Hello', '%15g', 0),
        print_header=True
    )
    assert(len(out.getvalue().strip().split('\n')) == 3)
