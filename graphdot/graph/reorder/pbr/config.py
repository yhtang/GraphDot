#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tempfile
import contextlib


default_config = {
    # general
    'mode': 'recursive',
    'objective': 'cut',
    'seed': -1,
    'cmaxnet': -1,
    'vcycles': 0,
    # main -> preprocessing -> min hash sparsifier
    'p-use-sparsifier': 'false',
    # main -> preprocessing -> community detection
    'p-detect-communities': 'false',
    # main -> coarsening
    'c-type': 'heavy_lazy',
    'c-s': 3.25,
    'c-t': 160,
    # main -> coarsening -> rating
    'c-rating-score': 'heavy_edge',
    'c-rating-use-communities': 'false',
    'c-rating-heavy_node_penalty': 'no_penalty',
    'c-rating-acceptance-criterion': 'best',
    'c-fixed-vertex-acceptance-criterion': 'fixed_vertex_allowed',
    # main -> initial partitioning
    # 'i-mode': 'recursive',
    # 'i-technique': 'multi',
    'i-mode': 'direct',
    'i-technique': 'flat',
    # initial partitioning -> coarsening
    # 'i-c-type': 'ml_style',
    # 'i-c-s': 1,
    # 'i-c-t': 150,
    # initial partitioning -> coarsening -> rating
    # 'i-c-rating-score': 'heavy_edge' ,
    # 'i-c-rating-use-communities': 'true',
    # 'i-c-rating-heavy_node_penalty': 'no_penalty',
    # 'i-c-rating-acceptance-criterion': 'best_prefer_unmatched',
    # 'i-c-fixed-vertex-acceptance-criterion': 'fixed_vertex_allowed',
    # initial partitioning -> initial partitioning
    'i-algo': 'pool',
    'i-runs': 20,
    # initial partitioning -> local search
    'i-r-type': 'twoway_fm',
    'i-r-runs': -1,
    'i-r-fm-stop': 'simple',
    'i-r-fm-stop-i': 50,
    # main -> local search
    'r-type': 'twoway_fm',
    'r-runs': -1,
    'r-fm-stop': 'simple',
    'r-fm-stop-alpha': 1,
    'r-fm-stop-i': 350,
}


@contextlib.contextmanager
def to_ini(config=None):
    '''Converts a dictionary-based configuration set to an .ini file so that
    it can be loaded by Kahypar's C extension module.

    Parameters
    ----------
    config: dict
        Key-value pairs as appears in
        https://github.com/kahypar/kahypar/tree/master/config.
    '''

    config = config if config is not None else default_config
    try:
        f = tempfile.NamedTemporaryFile(buffering=0)
        s = '\n'.join([f'{key}={value}' for key, value in config.items()])
        f.write(s.encode())
        yield f.name
    finally:
        pass
