#!/usr/bin/env python
# -*- coding: utf-8 -*-
import re


class Template:
    """
    Code generation helper
    """

    def __init__(self, template):
        try:
            self.template = open(template).read()
        except OSError:
            self.template = template

    def render(self, **substitutions):
        """
        substitutions: symbol=replacement

        If replacement is list-like, use trailing sequence of symbol match
        to join members; otherwise do plain substitution
        """

        text = self.template
        for symbol in substitutions:
            repl = substitutions[symbol]
            if isinstance(repl, (list, tuple)):
                pattern = r'\${%s([^}]*)}' % symbol
                text = re.sub(pattern, lambda m: m.group(1).join(repl), text)
            else:
                pattern = r'\${%s}' % symbol
                text = re.sub(pattern, repl, text)
        return text
