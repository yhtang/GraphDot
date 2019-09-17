import pytest
from ase.build import molecule
from ase.collections import g2
from graphdot.graph import Graph
from graphdot.kernel.molecular import Tang2019MolecularKernel


def test_molecular_kernel_on_organics(benchmark):

    graphs = [Graph.from_ase(molecule(name)) for name in g2.names
              if len(molecule(name)) > 1]
    kernel = Tang2019MolecularKernel()

    def fun(kernel, graphs):
        return kernel(graphs, nodal=True)

    g = benchmark.pedantic(fun, args=(kernel, graphs), iterations=3, rounds=3,
                           warmup_rounds=1)
