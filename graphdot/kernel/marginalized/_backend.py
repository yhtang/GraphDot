#!/usr/bin/env python
# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod


class Backend(ABC):
    @abstractmethod
    def __call__(self):
        pass
