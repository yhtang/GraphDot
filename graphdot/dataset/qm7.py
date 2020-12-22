#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import scipy.io
import pandas as pd
from ase import Atoms
from ._get import get


def QM7(
    download_url='http://quantum-machine.org/data/qm7.mat',
    local_filename='qm7.mat', overwrite=False, ase=False
):
    '''A 7165-molecule subset of the GDB-13 dataset. Molecules have up to 23
    total atoms and 7 heavy atoms. Atomization energies are computed at the
    Perdew-Burke-Ernzerhof hybrid functional (PBE0) level.

    References:
    - L. C. Blum, J.-L. Reymond, 970 Million Druglike Small Molecules for
    Virtual Screening in the Chemical Universe Database GDB-13,
    J. Am. Chem. Soc., 131:8732, 2009.
    - M. Rupp, A. Tkatchenko, K.-R. Müller, O. A. von Lilienfeld: Fast and
    Accurate Modeling of Molecular Atomization Energies with Machine
    Learning, Physical Review Letters, 108(5):058301, 2012

    Parameters
    ----------
    download_url: str
        URL to download the qm7.mat data file.
    local_filename: str
        Name for local storage of the data file.
    overwrite: bool
        Whether or not to overwrite the local file if one already exists.
    ase: bool
        Whether to create ASE Atoms objects from the given.

    Returns
    -------
    qm7: DataFrame
        A dataframe containing the data from QM7.
    '''
    try:
        mat = scipy.io.loadmat(
            get(download_url, local_filename, overwrite=overwrite)
        )
    except Exception as e:
        raise RuntimeError(
            f'Loading {local_filename} failed due to error: {e}.'
        )

    def _as_objects(array):
        out = np.empty(len(array), dtype=np.object)
        for i, element in enumerate(array):
            out[i] = element
        return out

    qm7 = pd.DataFrame(data=dict(
        columb_matrix=_as_objects(mat['X']),
        atomization_energy=mat['T'].ravel().astype(np.float),
        atomic_charge=_as_objects(mat['Z']),
        xyz=_as_objects(mat['R']),
        split=np.zeros(7165, dtype=np.int)
    ))

    for i, s in enumerate(mat['P']):
        qm7.loc[s, 'split'] = i

    if ase is True:
        qm7['atoms'] = qm7.apply(
            lambda row: Atoms(
                row.atomic_charge[row.atomic_charge != 0],
                row.xyz[row.atomic_charge != 0]
            ),
            axis=1
        )

    return qm7
