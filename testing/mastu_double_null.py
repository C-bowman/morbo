
from numpy import load, linspace

D = load('./testing_data/mastu_fiesta_equilibrium_5.npz')

R_psi = D['R']
z_psi = D['z']
psi_grid = D['psi']

from morbo.grid import GridGenerator
from morbo.tracing import Equilibrium

psi = Equilibrium(R_psi, z_psi, psi_grid, machine = 'mast-u')
psi.plot_equilibrium(flux_range = (0.95,1.15))
psi.plot_stationary_points()
# exit()






"""
Specify the flux axes for the 4 separatrix-bounded regions
"""
core_flux_grid = linspace(0.97,1,7)
pfr_flux_grid =  linspace(0.97,1,7)

outer_sol_flux_grid = linspace(1,1.03,7)[1:]
inner_sol_flux_grid = linspace(1,1.03,7)[1:]

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
                   machine = 'mast-u')

GG.plot_grids()