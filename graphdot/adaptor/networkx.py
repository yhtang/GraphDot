class NetworkXGraphAdaptor:
    
    def __init__(self):
        
        pass
    
    def __call__(self, G):
        
        vertices = [ (i, attr['label']) for i, attr in G.nodes.items() ]

        edges = []
        for i, neighbors in G.adjacency():
            for j, attr in neighbors.items():
                if j > i:
                    edges.append( ( i, j, attr['weight'], attr['label'] ) )

        return ( vertices, edges )