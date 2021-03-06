
from numpy import load, linspace

D = load('./testing_data/tcv_equil_64523_1100ms.npz')

R_psi = D['R']
z_psi = D['z']
psi_grid = D['psi']


from morbo.grid import GridGenerator
from morbo.equilibrium import Equilibrium
psi = Equilibrium(R_psi, z_psi, psi_grid)
psi.plot_stationary_points()


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














# core_flux_grid = [v for v in linspace(0.9,0.95,4)]
# core_flux_grid.extend([v for v in linspace(0.95,1.,8)[1:-1]])
#
# pfr_flux_grid = [v for v in linspace(0.9,0.95,4)]
# pfr_flux_grid.extend([v for v in linspace(0.95,1.,8)[1:-1]])
#
# outer_sol_flux_grid = [v for v in linspace(1,1.05,8)[1:]]
# outer_sol_flux_grid.extend([v for v in linspace(1.05,1.2,8)[1:]])
#
# inner_sol_flux_grid = [v for v in linspace(1,1.05,8)[1:]]
# inner_sol_flux_grid.extend([v for v in linspace(1.05,1.095,4)[1:]])
#
# outer_leg_distance_axis = concatenate([linspace(0, 0.9, 16), linspace(0.9, 1, 6)[1:]])
# inner_leg_distance_axis = concatenate([linspace(0, 0.65, 5), linspace(0.65, 1, 6)[1:]])
#
# outer_edge_distance_axis = linspace(0, 1, 30)
# inner_edge_distance_axis = linspace(0, 1, 30)