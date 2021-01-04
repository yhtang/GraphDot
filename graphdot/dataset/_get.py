#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import requests


def get(url, local_filename, overwrite=False, parser=None):
    '''Download a file from a given URL.'''

    if not os.path.exists(local_filename) or overwrite is True:
        r = requests.get(url)
        if r.status_code != 200:
            raise RuntimeError(
                f'Downloading from {url} failed with HTTP status '
                f'code {r.status_code}.'
            )
        open(local_filename, 'wb').write(r.content)

    if parser is not None:
        return parser(local_filename)
    else:
        return local_filename
