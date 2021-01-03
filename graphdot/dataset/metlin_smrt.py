#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
from ._get import get


def METLIN_SMRT(
    download_url='https://ndownloader.figshare.com/files/18130628',
    local_filename='SMRT_dataset.csv', overwrite=False
):
    '''Domingo-Almenara, X., Guijas, C., Billings, E. et al. The METLIN small
    molecule dataset for machine learning-based retention time prediction. Nat
    Commun 10, 5811 (2019). https://doi.org/10.1038/s41467-019-13680-7
    '''
    try:
        smrt = pd.read_csv(
            get(download_url, local_filename, overwrite=overwrite),
            sep=';'
        )
    except Exception as e:
        raise RuntimeError(
            f'Loading {local_filename} failed due to error: {e}.'
        )

    return smrt
