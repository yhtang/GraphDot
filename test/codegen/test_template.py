from graphdot.codegen import Template
import pytest


# prefixes = [
#     u'', u'a', u'0', u'~', u'!', u'@', u'#', u'$', u'
# ]

cases = [
    (r'${_1}', 'A', 'A'),
    (r'${_1}', '123', '123')
]


@pytest.mark.parametrize('case', cases)
def test_render_single(case):
    tpl, repl, result = case
    assert(Template(tpl).render(_1=repl) == result)


def test_render_list():
    assert(Template('${repl.}').render(repl=['a', 'b']) == 'a.b')
    assert(Template('${repl,}').render(repl=['a', 'b']) == 'a,b')
    assert(Template('${repl;}').render(repl=['a', 'b']) == 'a;b')
    assert(Template('${repl, }').render(repl=['a', 'b']) == 'a, b')
