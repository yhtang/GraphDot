#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
from ._get import get


def AMES(source, overwrite=False):
    '''Ames bacteria mutagenicity test data from multiple sources.

    Parameters
    ----------
    source: str or list of strs
        Available options:

        - '10.1021/ci300400a': around 8000 molecules, SMILES format.

    Returns
    -------
    data: DataFrame or collections of data frames.
        DataFrames containing the specified data.
    '''
    def _get(src):
        if src == '10.1021/ci300400a':
            return pd.read_excel(
                get('https://ndownloader.figshare.com/files/4108681',
                    'ci300400a_si_001.xls', overwrite=overwrite),
                sheet_name=None,
                header=1
            )
        else:
            raise RuntimeError(f'Unknown data source {src}.')

    if isinstance(source, str):
        return _get(source)
    else:
        return [_get(src) for src in source]
