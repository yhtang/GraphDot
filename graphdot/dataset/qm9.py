#!/usr/bin/env python
# -*- coding: utf-8 -*-
import io
import tarfile
import numpy as np
import pandas as pd
from ase import Atoms
from tqdm import tqdm
from ._get import get


def QM9(
    download_url='https://ndownloader.figshare.com/files/3195389',
    local_filename='dsgdb9nsd.xyz.tar.bz2', overwrite=False, ase=False
):
    '''Quantum chemistry structures and properties of 134 kilo molecules.

    References:

    - Ramakrishnan, Raghunathan, et al. "Quantum chemistry structures and
    properties of 134 kilo molecules." Scientific data 1.1 (2014): 1-7.

    Parameters
    ----------
    download_url: str
        URL to download the qm7.mat data file.
    local_filename: str
        Name for local storage of the data file.
    overwrite: bool
        Whether or not to overwrite the local file if one already exists.
    ase: bool
        Whether to create ASE Atoms objects from the dataset.

    Returns
    -------
    qm9: DataFrame
        A dataframe containing the data from QM9.
    '''
    try:
        f = get(download_url, local_filename)
    except Exception as e:
        raise RuntimeError(
            f'Acquiring {local_filename} failed due to error: {e}.'
        )

    data = []
    tf = tarfile.open(f, 'r:bz2')
    for xyz in tqdm(tf, total=133885):
        content = io.TextIOWrapper(tf.extractfile(xyz)).read()
        content = content.replace('*^', 'E')
        lines = content.split('\n')
        n_atoms = int(lines[0])
        fields = lines[1][4:].strip().split('\t')
        symbols, x, y, z, charges = zip(*[line.split('\t')
                                          for line in lines[2:n_atoms + 2]])
        data.append(tuple(
            # scalar properties
            [int(fields[0])] + [float(w) for w in fields[1:]] +
            # atomic coordinates
            [symbols, np.array([x, y, z]).T.tolist(), charges] +
            # vibrational frequencies
            [list(map(float, lines[n_atoms + 2].strip().split('\t')))] +
            # SMILES
            lines[n_atoms + 3].strip().split('\t') +
            # InChI
            lines[n_atoms + 4].strip().split('\t')
        ))

    qm9 = pd.DataFrame(
        data,
        columns=[
            'id', 'A', 'B', 'C', 'mu', 'alpha', 'e_HOMO', 'e_LUMO', 'e_gap',
            'R2', 'zpve', 'U0', 'U', 'H', 'G', 'Cv', 'symbols', 'xyz',
            'charges_mulliken', 'freq', 'smiles_gdb', 'smiles_opt',
            'inchi_gdb', 'inchi_opt'
        ]
    )

    if ase is True:
        qm9['atoms'] = qm9.apply(
            lambda row: Atoms(
                symbols=row.symbols,
                positions=row.xyz,
                charges=row.charges_mulliken
            ),
            axis=1
        )

    return qm9
