from numpy import load, linspace, meshgrid, array, dot, sqrt
import matplotlib.pyplot as plt

def dist(a,b):
    c = a - b
    return sqrt((c**2).sum())


D = load('tcv_equil_64523_1100ms.npz')

R_psi = D['R']
z_psi = D['z']
psi_grid = D['psi']
psi_shape = psi_grid.shape

print(R_psi.shape)

from morbo.grid import GridGenerator
from morbo.tracing import Equilibrium
psi = Equilibrium(R_psi, z_psi, psi_grid)


"""
Specify the flux axes for the 4 separatrix-bounded regions
"""
core_flux_grid = [v for v in linspace(0.9,0.95,4)]
core_flux_grid.extend([v for v in linspace(0.95,1.,8)[1:-1]])

pfr_flux_grid = [v for v in linspace(0.9,0.95,4)]
pfr_flux_grid.extend([v for v in linspace(0.95,1.,8)[1:-1]])

outer_sol_flux_grid = [v for v in linspace(1,1.05,8)[1:]]
outer_sol_flux_grid.extend([v for v in linspace(1.05,1.2,8)[1:]])

inner_sol_flux_grid = [v for v in linspace(1,1.05,8)[1:]]
inner_sol_flux_grid.extend([v for v in linspace(1.05,1.095,4)[1:]])


GG = GridGenerator(equilibrium = psi,
                   core_flux_grid = core_flux_grid,
                   pfr_flux_grid = pfr_flux_grid,
                   outer_sol_flux_grid = outer_sol_flux_grid,
                   inner_sol_flux_grid = inner_sol_flux_grid)