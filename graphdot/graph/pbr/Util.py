import sys
import os
import numpy as np
import logging

from scipy.sparse import coo_matrix


def tile_stats (A, tilesize, baseidx=0):
    '''
    A should be in COO format
    '''

    assert A.shape[0] == A.shape[1]

    nnzdict = {(i, j): 1 for (i, j) in  zip(A.row, A.col)}
    ntiles = int((A.shape[0] + tilesize - 1) / tilesize)
    nempty = 0
    for ti in range(ntiles):
        for tj in range(ntiles):
            begi = ti * tilesize + baseidx
            begj = tj * tilesize + baseidx
            hasnnz = False
            for i in range(tilesize):
                for j in range(tilesize):
                    if ((begi+i, begj+j) in nnzdict):
                        hasnnz = True
            if (not hasnnz):
                nempty += 1

    ntiles_tot = ntiles * ntiles
    ntiles_emp = nempty
    ntiles_occ = (ntiles * ntiles) - ntiles_emp
    if ((ntiles_occ)*tilesize*tilesize == 0):
        tile_fill = 0
    else:
        tile_fill = float(A.nnz) / ((ntiles_occ)*tilesize*tilesize)

    return ntiles_tot, ntiles_emp, ntiles_occ, tile_fill




def perm_sym (A, perm=None):
    '''
    Permutes A for a given permutation vector with symmetric permutation
    If perm is None, uses random permutation
    '''

    assert A.shape[0] == A.shape[1]

    if (perm is None):
        perm = np.random.permutation(range(A.shape[0]))

    invperm = [None] * A.shape[0]
    for i in range(A.shape[0]):
        invperm[perm[i]] = i
        
    prow  = [None] * A.nnz
    pcol  = [None] * A.nnz
    pdata = [None] * A.nnz
    idx = 0
    for (i, j, nnz) in zip(A.row, A.col, A.data):
        prow[idx] = invperm[i]
        pcol[idx] = invperm[j]
        pdata[idx] = nnz
        idx += 1
    
    return coo_matrix((np.array(pdata), (np.array(prow), np.array(pcol))),
                      shape=(A.shape[0], A.shape[1]))



def perm_gen (A, rperm=None, cperm=None):
    '''
    Permutes A for a given permutation vector with nonsymmetric permutation
    If perm is None, uses random permutation
    '''
    
    if (rperm == None):
        rperm = np.random.permutation(range(A.shape[0]))
    if (cperm == None):
        cperm = np.random.permutation(range(A.shape[1]))

    invrperm = [None] * A.shape[0]
    for i in range(A.shape[0]):
        invrperm[rperm[i]] = i
    invcperm = [None] * A.shape[1]
    for i in range(A.shape[1]):
        invcperm[cperm[i]] = i
        
    prow  = [None] * A.nnz
    pcol  = [None] * A.nnz
    pdata = [None] * A.nnz
    idx = 0
    for (i, j, nnz) in zip(A.row, A.col, A.data):
        prow[idx] = invrperm[i]
        pcol[idx] = invcperm[j]
        pdata[idx] = nnz
        idx += 1
    
    return coo_matrix((np.array(pdata), (np.array(prow), np.array(pcol))),
                      shape=(A.shape[0], A.shape[1]))



def readtxtmat (filename):
    '''
    Each line contains a row
    '''
    
    lines = []
    nnzs = 0
    f = open(filename)
    for line in f:
        tmp = line.split()
        for nnz in tmp:
            if (nnz != '0'):
                nnzs += 1
        lines.append(tmp)
    f.close()

    row = [None] * nnzs
    col = [None] * nnzs
    data = [None] * nnzs
    idx = 0
    for i in range(len(lines)):
        assert len(lines[i]) == len(lines)
        for j in range(len(lines[i])):
            if (lines[i][j] != '0'):
                row[idx] = i
                col[idx] = j
                data[idx] = float(lines[i][j])
                idx += 1
    A = coo_matrix((np.array(data), (np.array(row), np.array(col))),
                   shape=(len(lines), len(lines)))

    return A



def writetxtmat (A, fname):
    '''
    Input should in COO fomat
    '''

    f = open(fname, 'w')
    nnzdict = {(i, j): k for (i, j, k) in  zip(A.row, A.col, A.data)}
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if ((i,j) not in nnzdict):
                f.write('0 ')
            else:
                f.write(str(nnzdict[(i,j)]) + ' ')
        f.write('\n')
    f.close()

    return



def read2dmat (path):
    '''
    Stores the read matrix in 2D matrix
    '''
    
    f = open(path)
    data = []
    
    for line in f:
        tmp = line.split()
        arr = []
        for t in tmp:
            if t == '0':
                arr.append(0)
            else:
                arr.append(1)
        data.append(arr)
    
    f.close()
    
    for arr in data:
        assert len(arr) == len(data)
    
    return data



def writemtxmat (A, fname):
    '''
    Input should be in COO format

    Assumes matrix is symmetric if it is symmetric patternwise
    '''

    issym = True
    nnzdict = {}
    for (i,j) in zip(A.row, A.col):
        tmpi = min(i,j)
        tmpj = max(i,j)
        if (tmpi, tmpj) not in nnzdict:
            nnzdict[(tmpi,tmpj)] = 1
        else:
            nnzdict[(tmpi,tmpj)] += 1

    for (k,v) in nnzdict.iteritems():
        if (k[0] != k[1] and v == 1):
            issym = False
            break
        if (k[0] == k[1]):
            assert v == 1

    f = open(fname, 'w')
    if (issym):
        logging.info('writing symmetric matrix to file ' + fname)
        f.write('%%MatrixMarket matrix coordinate pattern symmetric\n')
        f.write('%d %d %d\n' % (A.shape[0], A.shape[1], len(nnzdict)))
        for (k, v) in nnzdict.iteritems():
            f.write('%d %d\n' % (k[0]+1, k[1]+1))
    else:
        logging.info('writing general matrix to file ' + fname)
        f.write('%%MatrixMarket matrix coordinate pattern general\n')
        f.write('%d %d %d\n' % (A.shape[0], A.shape[1], A.nnz))
        for (i,j) in zip(A.row, A.col):
            f.write('%d %d\n' % (i+1, j+1))

    f.close()

    return




def readpfile (fname):

    pvec = []
    f = open(fname)
    for line in f:
        pvec.append(int(line))
    f.close()
    return pvec
