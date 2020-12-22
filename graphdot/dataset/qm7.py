#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import numpy as np
import scipy.io
import pandas as pd
import requests
from ase import Atoms


def QM7(
    download_url='http://quantum-machine.org/data/qm7.mat',
    local_filename='qm7.mat', overwrite=False, ase=False
):
    if not os.path.exists(local_filename) or overwrite is True:
        r = requests.get(download_url)
        if r.status_code != 200:
            raise RuntimeError(
                f'Downloading from {download_url} failed with HTTP status '
                f'code {r.status_code}.'
            )
        open(local_filename, 'wb').write(r.content)

    try:
        mat = scipy.io.loadmat(local_filename)
    except Exception as e:
        raise RuntimeError(
            f'Local file {local_filename} exists but cannot be loaded due '
            f'to error: {e}.'
        )

    def _as_objects(array):
        out = np.empty(len(array), dtype=np.object)
        for i, element in enumerate(array):
            out[i] = element
        return out

    df = pd.DataFrame(data=dict(
        columb_matrix=_as_objects(mat['X']),
        atomization_energy=mat['T'].ravel().astype(np.float),
        atomic_charge=_as_objects(mat['Z']),
        xyz=_as_objects(mat['R']),
        split=np.zeros(7165, dtype=np.int)
    ))

    for i, s in enumerate(mat['P']):
        df.loc[s, 'split'] = i

    if ase is True:
        df['atoms'] = df.apply(
            lambda row: Atoms(
                row.atomic_charge[row.atomic_charge != 0],
                row.xyz[row.atomic_charge != 0]
            ),
            axis=1
        )

    return df
