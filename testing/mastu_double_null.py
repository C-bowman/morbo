
from numpy import load, linspace

D = load('./testing_data/mastu_fiesta_equilibrium_5.npz')

R_psi = D['R']
z_psi = D['z']
psi_grid = D['psi']

from morbo.grid import GridGenerator
from morbo.equilibrium import Equilibrium

psi = Equilibrium(R_psi, z_psi, psi_grid, machine = 'mast-u')
psi.plot_equilibrium(flux_range = (0.95,1.15))
psi.plot_stationary_points()
# exit()


from numpy import concatenate
def midpoints(x):
    return 0.5*(x[1:]+x[:-1])

"""
Specify the flux axes for the 4 separatrix-bounded regions
"""
core_flux_grid = linspace(0.96,1,5)

pfr_flux_grid =  linspace(0.90,1,11)
pfr_flux_grid = concatenate([pfr_flux_grid, midpoints(pfr_flux_grid[-3:])])
pfr_flux_grid.sort()

outer_sol_flux_grid = linspace(1,1.20,21)
outer_sol_flux_grid = concatenate([outer_sol_flux_grid, midpoints(outer_sol_flux_grid[:5])])
outer_sol_flux_grid.sort()
outer_sol_flux_grid = outer_sol_flux_grid[1:]

inner_sol_flux_grid = linspace(1,1.03,4)[1:]

outer_leg_distance_axis = linspace(0, 1, 30)
outer_leg_distance_axis = concatenate([outer_leg_distance_axis, midpoints(outer_leg_distance_axis[-3:])])
outer_leg_distance_axis.sort()

inner_leg_distance_axis = linspace(0, 1, 6)

outer_edge_distance_axis = linspace(0, 1, 10)
inner_edge_distance_axis = linspace(0, 1, 10)




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

from mesh_tools.mesh import TriangularMesh, Triangle
def triangles_from_grid(R,z,separatrix_ind):
    # build a mesh by splitting SOLPS grid cells
    triangles = []
    for i in range(R.shape[0]-1):
        for j in range(R.shape[1]-1):
            p1 = (R[i,j], z[i,j])
            p2 = (R[i+1, j], z[i+1, j])
            p3 = (R[i, j+1], z[i, j+1])
            p4 = (R[i+1, j+1], z[i+1, j+1])

            if j >= separatrix_ind:
                triangles.append(Triangle(p1, p2, p3))
                triangles.append(Triangle(p2, p3, p4))
            else:
                triangles.append(Triangle(p1, p2, p4))
                triangles.append(Triangle(p1, p3, p4))
    return triangles

grid = GG.outer_leg_grid
grid.R = grid.R[9:,:]
grid.z = grid.z[9:,:]
grid.distance = grid.distance[9:]


triangles = triangles_from_grid(grid.R, grid.z, 50) #grid.lcfs_index)
mesh = TriangularMesh(triangles = triangles)
print(mesh)
mesh.save('superx_field_mesh_v851.npz')

from morbo.operators import parallel_derivative, perpendicular_derivative

# para_dict = parallel_derivative(grid)
perp_dict = perpendicular_derivative(grid)

import matplotlib.pyplot as plt
from morbo.boundary import mastu_boundary
mesh.draw(plt, color = 'blue')
plt.plot(*mastu_boundary(), c = 'black')
plt.axis('equal')
plt.xlabel('R (m)')
plt.ylabel('z (m)')
plt.show()