#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import pytest
from graphdot.codegen import Template

_pwd = os.path.dirname(os.path.abspath(__file__))

single_substitution_cases = [
    (r'${_1}', 'A', 'A'),
    (r'${_1}', '123', '123'),
]


@pytest.mark.parametrize('case', single_substitution_cases)
def test_render_single(case):
    tpl, repl, result = case
    assert(Template(tpl).render(_1=repl) == result)


list_substitution_cases = [
    (['.', ',', ';', ':', '|', '&', '^'], []),
    (['.', ',', ';', ':', '|', '&', '^'], ['a']),
    (['.', ',', ';', ':', '|', '&', '^'], ['a', 'b']),
    (['.', ',', ';', ':', '|', '&', '^'], ['a', 'b', '123']),
]


@pytest.mark.parametrize('case', list_substitution_cases)
def test_render_list(case):
    separators, repls = case
    for sep in separators:
        assert(Template(r'${key%s}' % sep).render(key=repls)
               == sep.join(repls))


file_substitution_cases = [
    ('', '', '', ' +  = \n'),
    ('', '', 'c', ' +  = c\n'),
    ('a', 'b', '', 'a + b = \n'),
    ('a', 'b', 'c', 'a + b = c\n'),
]


@pytest.mark.parametrize('case', file_substitution_cases)
def test_render_file(case):
    key1, key2, val, result = case
    assert(Template(os.path.join(_pwd, 'test_template.tpl')).render(key1=key1,
                                                                    key2=key2,
                                                                    val=val)
           == result)
