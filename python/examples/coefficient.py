#!/usr/bin/env python3

import argparse as ap
from mpi4py import MPI
from petsc4py import PETSc
import cudolfinx as cufem
from dolfinx import fem, mesh
from dolfinx.fem import petsc as fe_petsc
import numpy as np
import time


#domain = mesh.create_unit_square(MPI.COMM_WORLD, 8, 8, mesh.CellType.quadrilateral)
domain = mesh.create_unit_cube(MPI.COMM_WORLD, 20, 20, 20, mesh.CellType.tetrahedron)
V = fem.functionspace(domain, ("Lagrange", 4))
V2 = fem.functionspace(domain, ("Lagrange", 3))
u = fem.Function(V)
u_true = fem.Function(V)
u2 = fem.Function(V2)

u2.interpolate(lambda x: 1 + 0.1*x[0]**2 + 0.2*x[1]**2 + 0.3*x[2]**2)

if __name__ == "__main__":
    niter = 5

    u_true.interpolate(u2)
    start = time.perf_counter()
    for _ in range(niter):
        u_true.interpolate(u2)
    
    end = time.perf_counter()
    print(f"CPU Time: {1e3*(end-start)/niter:.2f} ms")

    coeff = cufem.Coefficient(u)
    coeff.interpolate(u2) # initialize

    start = time.perf_counter()
    for _ in range(niter):
        coeff.interpolate(u2)

    end = time.perf_counter()

    print(f"GPU Time: {1e3*(end-start)/niter:.2f} ms")

    assert np.allclose(u_true.x.array, coeff.values(), rtol=1e-12)
    print("PASSED")
