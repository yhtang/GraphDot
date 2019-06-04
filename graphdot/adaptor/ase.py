import numpy
import sklearn.metrics

class ASEAtomsAdaptor:

    def __init__(self, zoom=5.0 ):
        
        self.zoom = zoom
        
    def __call__(self, atoms):
        
        vertices = [ (i, n) for i, n in enumerate( atoms.get_atomic_numbers() ) ]
        
        edges = []
        
        D = sklearn.metrics.pairwise_distances( atoms.get_positions() )
        
        for i, row in enumerate( D ):
            for j, d in enumerate( row[i+1:] ):
                if d < self.zoom:
                    w = ( 1 - d / self.zoom )**2
                    edges.append( ( i, i+j, w, d ) )
                    
        return ( vertices, edges )
