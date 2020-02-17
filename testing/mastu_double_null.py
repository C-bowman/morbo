
from numpy import load, linspace, meshgrid

D = load('./testing_data/mastu_fiesta_equilibrium_5.npz')

R_psi = D['R']
z_psi = D['z']
psi_grid = D['psi']
psi_shape = psi_grid.shape
print(R_psi.shape, z_psi.shape, psi_grid.shape)

from morbo.grid import GridGenerator
from morbo.tracing import Equilibrium

import matplotlib.pyplot as plt

R_mesh, z_mesh = meshgrid(R_psi, z_psi)


plt.scatter(R_mesh, z_mesh, c = psi_grid, marker = '.')
plt.axis('equal')
plt.tight_layout()
plt.show()

psi = Equilibrium(R_psi, z_psi, psi_grid)
psi.plot_stationary_points()
exit()






"""
Specify the flux axes for the 4 separatrix-bounded regions
"""
core_flux_grid = linspace(0.91,1,7)
pfr_flux_grid =  linspace(0.91,1,7)

outer_sol_flux_grid = linspace(1,1.09,7)[1:]
inner_sol_flux_grid = linspace(1,1.09,7)[1:]

outer_leg_distance_axis = linspace(0, 1, 16)
inner_leg_distance_axis = linspace(0, 1, 7)

outer_edge_distance_axis = linspace(0, 1, 40)
inner_edge_distance_axis = linspace(0, 1, 30)




GG = GridGenerator(equilibrium = psi,
                   core_flux_grid = core_flux_grid,
                   pfr_flux_grid = pfr_flux_grid,
                   outer_sol_flux_grid = outer_sol_flux_grid,
                   inner_sol_flux_grid = inner_sol_flux_grid,
                   inner_leg_distance_axis = inner_leg_distance_axis,
                   outer_leg_distance_axis = outer_leg_distance_axis,
                   inner_edge_distance_axis = inner_edge_distance_axis,
                   outer_edge_distance_axis = outer_edge_distance_axis,
                   machine = 'tcv')

GG.plot_grids()