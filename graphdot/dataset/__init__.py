#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
from ._get import get
from .qm7 import QM7


__all__ = ['get', 'QM7', 'METLIN_SMRT', 'AMES']


def METLIN_SMRT(
    download_url='https://ndownloader.figshare.com/files/18130628',
    local_filename='SMRT_dataset.csv', overwrite=False
):
    '''Domingo-Almenara, X., Guijas, C., Billings, E. et al. The METLIN small
    molecule dataset for machine learning-based retention time prediction. Nat
    Commun 10, 5811 (2019). https://doi.org/10.1038/s41467-019-13680-7
    '''
    return get(
        download_url, local_filename, overwrite=overwrite,
        parser=lambda f: pd.read_csv(f, sep=';')
    )


def AMES(overwrite=False):
    '''Ames bacteria mutagenicity test data from multiple sources.
    Data source: 10.1021/ci300400a, around 8000 molecules, SMILES format.
    '''
    return get(
        'https://ndownloader.figshare.com/files/4108681',
        'ci300400a_si_001.xls',
        overwrite=overwrite,
        parser=lambda f: pd.read_excel(f, sheet_name=None, header=1)
    )
