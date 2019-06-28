#!/usr/bin/env python
# -*- coding: utf-8 -*-
import re
import os

__all__ = ['Template']


class Template:
    """
    Code generation helper
    """

    def __init__(self, template):
        if os.path.isfile(template):
            self.template = open(template).read()
        else:
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
