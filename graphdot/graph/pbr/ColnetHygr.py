import sys

from Hygr import Hygr


class ColnetHygr(Hygr):

    '''
    Standard colnet hypergraph from matrix market file, or any other means.
    Supports symmetric and non-symmetric matrices.
    '''


    def __init__ (self, unitVertexWeights=False):
        super(ColnetHygr, self).__init__(unitVertexWeights)

        return



    def createFromPairs (self, row_ids, col_ids, nr, nc, is_sym=True):

        if (not is_sym):
            sys.exit(1)

        self._nverts = nr
        self._nnets = nc
        self._xpins = [0] * (self._nnets+2) # final element is auxiliary
        self._nwghts = [1] * self._nnets
        self._cwghts = [0] * self._nverts
        self._nconst = 1

        for (i,j) in zip(row_ids, col_ids):
            if (i != j):
                self._xpins[j+2] += 1
                self._cwghts[i] += 1

        # enforce diagonals - they are needed in the algorithm
        for j in range(nc):
            self._xpins[j+2] += 1

        for net in range(self._nnets+1):
            self._xpins[net+1] += self._xpins[net]

        self._npins = self._xpins[self._nnets+1]
        self._pins = [None] * self._npins

        for j in range(nc):
            self._pins[self._xpins[j+1]] = j
            self._xpins[j+1] += 1

        for (i,j) in zip(row_ids, col_ids):
            if (i != j):
                self._pins[self._xpins[j+1]] = i
                self._xpins[j+1] += 1

        if (self._unitVertexWeights):
            for v in range(self._nverts):
                self._cwghts[v] = 1

        return



    def write (self, outfname):

        f = open(outfname, 'w')
        f.write('% colnet hypergraph\n')
        super(ColnetHygr, self)._write(f)
        f.close()

        return



    def print_hygr (self):

        sys.stdout.write('printing colnet hypergraph\n')
        super(ColnetHygr, self).print_hygr()

        return
    
        
    
