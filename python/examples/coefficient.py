#!/usr/bin/env python3

import argparse as ap
from mpi4py import MPI
from petsc4py import PETSc
import cudolfinx as cufem
from dolfinx import fem, mesh
from dolfinx.fem import petsc as fe_petsc
import numpy as np


domain = mesh.create_unit_square(MPI.COMM_WORLD, 8, 8, mesh.CellType.quadrilateral)
V = fem.functionspace(domain, ("Lagrange", 1))
u = fem.Function(V)

if __name__ == "__main__":
    coeff = cufem.Coefficient(u)
