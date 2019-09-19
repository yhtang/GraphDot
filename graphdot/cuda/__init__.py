#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pycuda.driver

try:
    pycuda.driver.init()
except Exception as e:
    raise RuntimeWarning('PyCUDA initialization failed, message: ' + str(e))
