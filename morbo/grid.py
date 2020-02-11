
from numpy import array, dot, sqrt, sign, zeros, load, linspace, pi, sin, cos, concatenate
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import differential_evolution
from mesh_tools.vessel_boundaries import tcv_baffled_boundary

import matplotlib.pyplot as plt


from morbo.tracing import EqTracer

D = load('tcv_equil_64523_1100ms.npz')

R_psi = D['R']
z_psi = D['z']
psi_grid = D['psi']
psi_shape = psi_grid.shape






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

psi_axis_lcfs_inds = [len(pfr_flux_grid), len(core_flux_grid), len(pfr_flux_grid), len(core_flux_grid)]


"""
now use these to build the flux axes for the legs / edges
"""
ol_psi_axis = []
ol_psi_axis.extend(pfr_flux_grid)
ol_psi_axis.append(1.)
ol_psi_axis.extend(outer_sol_flux_grid)

oe_psi_axis = []
oe_psi_axis.extend(core_flux_grid)
oe_psi_axis.append(1.)
oe_psi_axis.extend(outer_sol_flux_grid)

ie_psi_axis = []
ie_psi_axis.extend(core_flux_grid)
ie_psi_axis.append(1.)
ie_psi_axis.extend(inner_sol_flux_grid)

il_psi_axis = []
il_psi_axis.extend(pfr_flux_grid)
il_psi_axis.append(1.)
il_psi_axis.extend(inner_sol_flux_grid)

psi_axes = [il_psi_axis, ie_psi_axis, ol_psi_axis, oe_psi_axis]


"""
specify the poloidal distance grids
"""
ol_distance_axis = concatenate([linspace(0,0.9,16), linspace(0.9,1,6)[1:]])
il_distance_axis = concatenate([linspace(0,0.65,5), linspace(0.65,1,6)[1:]])

oe_distance_axis = linspace(0,1,12)
ie_distance_axis = linspace(0,1,12)

distance_axes = [il_distance_axis, ie_distance_axis, ol_distance_axis, oe_distance_axis]


"""
build the R/z grids for the legs / edges
"""
R_ol = zeros([len(ol_distance_axis), len(ol_psi_axis)])
z_ol = zeros([len(ol_distance_axis), len(ol_psi_axis)])

R_oe = zeros([len(oe_distance_axis), len(oe_psi_axis)])
z_oe = zeros([len(oe_distance_axis), len(oe_psi_axis)])

R_il = zeros([len(il_distance_axis), len(il_psi_axis)])
z_il = zeros([len(il_distance_axis), len(il_psi_axis)])

R_ie = zeros([len(ie_distance_axis), len(ie_psi_axis)])
z_ie = zeros([len(ie_distance_axis), len(ie_psi_axis)])

R_grids = [R_il, R_ie, R_ol, R_oe]
z_grids = [z_il, z_ie, z_ol, z_oe]


"""
normalise the psi
"""
psi = EqTracer(R_psi, z_psi, psi_grid)

xpt_location_bounds = [(0.6,1.0), (-0.5,-0.2)]
xpt_result = differential_evolution(psi.nabla, bounds = xpt_location_bounds)
xpt = xpt_result.x

axis_location_bounds = [(0.7,1.1), (-0.2, 0.4)]
axis_result = differential_evolution(psi, bounds = axis_location_bounds)
axis = axis_result.x

psi_xpt = psi(xpt)
psi_axis = psi(axis)

psi_grid = (psi_grid - psi_axis) / (psi_xpt - psi_axis)
psi = EqTracer(R_psi, z_psi, psi_grid)


n_R, n_z = 128, 256
R_interp = linspace(R_psi.min(), R_psi.max(), n_R)
z_interp = linspace(z_psi.min(), z_psi.max(), n_z)
psi_interp = zeros([n_R, n_z])
nabla = zeros([n_R, n_z])

for i in range(n_R):
    for j in range(n_z):
        v = [R_interp[i], z_interp[j]]
        psi_interp[i,j] = psi(v)


"""
Find the 4 directions pointing along the separatrix from the x-point
"""
theta = linspace(0, 2*pi, 720)
lcfs_deviation = (1-psi.psi_spline.ev(xpt[0] + 0.015*cos(theta), xpt[1] + 0.01*sin(theta)))**2
lcfs_angle = theta[lcfs_deviation.argmin()]

cardinals = [0, 0.5*pi, pi, 1.5*pi]
lcfs_directions = [ array([cos(lcfs_angle+t), sin(lcfs_angle+t)]) for t in cardinals ]

# get the direction vector from the lower x-point to the axis
axis_drn = array([axis[0]-xpt[0], axis[1]-xpt[1]])
axis_drn /= sqrt(dot(axis_drn,axis_drn))
# also get the outboard direction
outboard_drn = array([1, 0])
# first sort by leg / edge region
lcfs_directions = sorted(lcfs_directions, key = lambda x : sign(dot(x,axis_drn)))
# then sort by inner / outer
lcfs_directions = sorted(lcfs_directions, key = lambda x : sign(dot(x,outboard_drn)))
# this means the order should be inner leg, inner edge, outer leg, outer edge
il_drn, ie_drn, ol_drn, oe_drn = lcfs_directions



"""
form the grid boundaries
"""
# create starting points for tracing the grid boundaries
gap = 0.01
for R,z,drn,psi_ax in zip(R_grids, z_grids, lcfs_directions, psi_axes):
    start = psi.follow_gradient(xpt + gap*drn, target_psi=1.)
    # trace outer-leg boundary
    for i,p in enumerate(psi_ax):
        x = psi.follow_gradient(start, target_psi = p)
        R[0,i] = x[0]
        z[0,i] = x[1]



"""
match grids at x-point
"""
for R,z,m in zip(R_grids, z_grids, psi_axis_lcfs_inds):
    R[0,m] = xpt[0]
    z[0,m] = xpt[1]



"""
match the outer-leg / outer-edge boundary
"""
outer_sol_boundary_R = 0.5*(R_ol[0,len(pfr_flux_grid):] + R_oe[0,len(core_flux_grid):])
R_ol[0,len(pfr_flux_grid):] = outer_sol_boundary_R
R_oe[0,len(core_flux_grid):] = outer_sol_boundary_R

outer_sol_boundary_z = 0.5*(z_ol[0,len(pfr_flux_grid):] + z_oe[0,len(core_flux_grid):])
z_ol[0,len(pfr_flux_grid):] = outer_sol_boundary_z
z_oe[0,len(core_flux_grid):] = outer_sol_boundary_z



"""
match the inner-leg / inner-edge boundary
"""
inner_sol_boundary_R = 0.5*(R_il[0,len(pfr_flux_grid):]+R_ie[0,len(core_flux_grid):])
R_il[0,len(pfr_flux_grid):] = inner_sol_boundary_R
R_ie[0,len(core_flux_grid):] = inner_sol_boundary_R

inner_sol_boundary_z = 0.5*(z_il[0,len(pfr_flux_grid):]+z_ie[0,len(core_flux_grid):])
z_il[0,len(pfr_flux_grid):] = inner_sol_boundary_z
z_ie[0,len(core_flux_grid):] = inner_sol_boundary_z



"""
match the inner-leg / outer-leg boundary
"""
pfr_boundary_R = 0.5*(R_il[0,:len(pfr_flux_grid)]+R_ol[0,:len(pfr_flux_grid)])
R_il[0,:len(pfr_flux_grid)] = pfr_boundary_R
R_ol[0,:len(pfr_flux_grid)] = pfr_boundary_R

pfr_boundary_z = 0.5*(z_il[0,:len(pfr_flux_grid)]+z_ol[0,:len(pfr_flux_grid)])
z_il[0,:len(pfr_flux_grid)] = pfr_boundary_z
z_ol[0,:len(pfr_flux_grid)] = pfr_boundary_z



"""
match the inner-edge / outer-edge boundary
"""
core_boundary_R = 0.5*(R_ie[0,:len(core_flux_grid)]+R_oe[0,:len(core_flux_grid)])
R_ie[0,:len(core_flux_grid)] = core_boundary_R
R_oe[0,:len(core_flux_grid)] = core_boundary_R

core_boundary_z = 0.5*(z_ie[0,:len(core_flux_grid)]+z_oe[0,:len(core_flux_grid)])
z_ie[0,:len(core_flux_grid)] = core_boundary_z
z_oe[0,:len(core_flux_grid)] = core_boundary_z



from mesh_tools.mesh import Polygon
bound_poly = Polygon(*tcv_baffled_boundary())
conditions = [bound_poly.is_inside, lambda x : dist(xpt,x) < 0.25, bound_poly.is_inside, lambda x : dist(xpt,x) < 0.25]
# conditions = [bound_poly.is_inside, lambda x : (x[0] < axis[0])|(x[1] < axis[1]), bound_poly.is_inside, lambda x : (x[0] > axis[0])|(x[1] < axis[1])]
trace_directions = [-1, 1, 1, -1]



"""
trace the grids using total poloidal distance
"""
data = [R_grids, z_grids, lcfs_directions, psi_axes, distance_axes, psi_axis_lcfs_inds, conditions, trace_directions]
for R, z, lcfs_drn, psi_ax, dist_ax, m, cond, trace_drn in zip(*data):

    total_poloidal_distance = zeros(len(psi_ax))
    for i in range(len(psi_ax)):
        if i != m:
            v = array([R[0,i], z[0,i]])
            x0, distance = psi.follow_surface_while(v, cond, direction=trace_drn)
            total_poloidal_distance[i] = distance
        else:
            v = array([R[0,i] + 0.001*lcfs_drn[0], z[0,i] + 0.001*lcfs_drn[1]])
            x0, distance = psi.follow_surface_while(v, cond, direction=trace_drn)
            total_poloidal_distance[i] = distance + 0.001

        R[-1,i] = x0[0]
        z[-1,i] = x0[1]

    for j in range(len(psi_ax)):
        gaps = dist_ax * total_poloidal_distance[j]
        for i,d in enumerate(gaps[1:-1]):
            if j != m and i != 0:
                v = array([R[0,j], z[0,j]])
                x0 = psi.follow_surface(v, d, direction=trace_drn)
            else:
                v = array([R[0,j] + 0.001*lcfs_drn[0], z[0,j] + 0.001*lcfs_drn[1]])
                x0 = psi.follow_surface(v, d, direction=trace_drn)

            R[i+1,j] = x0[0]
            z[i+1,j] = x0[1]




"""
trace the grids using grad-psi
"""
R_grids_orth = [R.copy() for R in R_grids]
z_grids_orth = [z.copy() for z in z_grids]

data = [R_grids_orth, z_grids_orth, lcfs_directions, psi_axes, distance_axes, psi_axis_lcfs_inds, conditions, trace_directions]
for R, z, lcfs_drn, psi_ax, dist_ax, m, cond, trace_drn in zip(*data):

    total_poloidal_distance = zeros(len(psi_ax))
    for i in range(len(psi_ax)):
        if i != m:
            v = array([R[0,i], z[0,i]])
            x0, distance = psi.follow_surface_while(v, cond, direction=trace_drn)
            total_poloidal_distance[i] = distance
        else:
            v = array([R[0,i] + 0.001*lcfs_drn[0], z[0,i] + 0.001*lcfs_drn[1]])
            x0, distance = psi.follow_surface_while(v, cond, direction=trace_drn)
            total_poloidal_distance[i] = distance + 0.001

        R[-1,i] = x0[0]
        z[-1,i] = x0[1]

    # now trace all the points in the separatrix
    gaps = dist_ax * total_poloidal_distance[m]
    for i,d in enumerate(gaps[1:]):
        v = array([R[0,m] + 0.001*lcfs_drn[0], z[0,m] + 0.001*lcfs_drn[1]])
        x0 = psi.follow_surface(v, d, direction=trace_drn)

        R[i+1,m] = x0[0]
        z[i+1,m] = x0[1]

    # now
    for j in range(len(psi_ax)):
        gaps = dist_ax * total_poloidal_distance[j]
        for i,d in enumerate(gaps[1:]):
            if j != m:
                v = array([R[i+1,m], z[i+1,m]])
                x0 = psi.follow_gradient(v, psi_ax[j])
                R[i+1,j] = x0[0]
                z[i+1,j] = x0[1]





"""
interpolate between the orthogonal / non-orthogonal grids
"""
# smoothly interpolate between orthogonal and non-orthogonal for the inner leg
from numpy import exp
smoothing = exp(-0.5*((il_distance_axis-1.)/0.25)**4)

R_il = R_grids[0]*smoothing[:,None] + (1-smoothing)[:,None]*R_grids_orth[0]
z_il = z_grids[0]*smoothing[:,None] + (1-smoothing)[:,None]*z_grids_orth[0]

# use the purely orthogonal grids for the edges
R_ie = R_grids_orth[1]
z_ie = z_grids_orth[1]

R_oe = R_grids_orth[3]
z_oe = z_grids_orth[3]

# for the outer leg, just replace the last row to be non-orthogonal
R_ol = R_grids_orth[2]
z_ol = z_grids_orth[2]
R_ol[-1,:] = R_grids[2][-1,:]
z_ol[-1,:] = z_grids[2][-1,:]






R_grids = [R_il, R_ie, R_ol, R_oe]
z_grids = [z_il, z_ie, z_ol, z_oe]




fig = plt.figure( figsize=(6,8))
ax1 = fig.add_subplot(111)
ax1.contour(R_interp,z_interp,psi_interp.T, levels = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3])
# plt.axis('equal')
# plt.tight_layout()
# plt.show()
# ax1.plot(*xpt, marker = 'x', markersize = 8, c = 'red')
# ax1.plot([xpt[0], xpt[0]+0.02*V[0,0]], [xpt[1], xpt[1]+0.02*V[0,1]], c = 'red')
# ax1.plot([xpt[0], xpt[0]+0.02*V[1,0]], [xpt[1], xpt[1]+0.02*V[1,1]], c = 'red')

plt.plot(*tcv_baffled_boundary(), lw = 2, c = 'black')
def plot_grid(ax, R, z, c):
    for i in range(R.shape[0]):
        ax.plot(R[i,:], z[i,:], c = c)
    for i in range(R.shape[1]):
        ax.plot(R[:, i], z[:, i], c = c)

plot_grid(ax1, R_ol, z_ol, 'blue')
plot_grid(ax1, R_oe, z_oe, 'red')
plot_grid(ax1, R_ie, z_ie, 'green')
plot_grid(ax1, R_il, z_il, 'purple')

# plt.tight_layout()
ax1.axis('equal')
ax1.set_xlim([R_psi.min(), R_psi.max()])
ax1.set_ylim([z_psi.min(), None])
# ax1.set_xlim([R.min(), R.max()])

plt.show()


from mesh_tools.mesh import TriangularMesh, Triangle
def triangles_from_grid(R,z,separatrix_ind):
    # build a mesh by splitting SOLPS grid cells
    triangles = []
    for i in range(R.shape[0]-1):
        for j in range(R.shape[1]-1):
            p1 = (R[i, j], z[i, j])
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


triangles = []
triangles.extend( triangles_from_grid(R_ol, z_ol, len(pfr_flux_grid)) )
triangles.extend( triangles_from_grid(R_il, z_il, len(pfr_flux_grid)) )
triangles.extend( triangles_from_grid(R_oe, z_oe, len(core_flux_grid)) )
triangles.extend( triangles_from_grid(R_ie, z_ie, len(core_flux_grid)) )

mesh = TriangularMesh(triangles = triangles)
#
fig = plt.figure( figsize=(6,8))
ax1 = fig.add_subplot(111)
# ax1.contour(R_interp,z_interp,psi_interp.T, levels = [0.7, 0.8, 0.9, 1.1, 1.2, 1.3])
mesh.draw(ax1, color = 'red')
# plt.plot([v[0] for v in mesh.vertices], [v[1] for v in mesh.vertices], ls = 'none', marker = '.', c = 'blue')

plt.plot(*tcv_baffled_boundary(), lw = 2, c = 'black')

ax1.axis('equal')
ax1.set_ylim([-0.8,-0.1])
ax1.set_xlim([0.6,1.1])
plt.tight_layout()
plt.show()
#
print(mesh)

mesh.save('tcv_field_mesh_64523_1100ms.npz')


# mesh.draw_surface([psi(v) for v in mesh.vertices])









flux_surfaces_R = []
flux_surfaces_z = []
"""
Parallel direction arrays
"""
pfr_surfaces_R = [concatenate([R_ol[::-1,i], R_il[1:,i]]) for i in range(len(pfr_flux_grid))]
pfr_surfaces_z = [concatenate([z_ol[::-1,i], z_il[1:,i]]) for i in range(len(pfr_flux_grid))]

core_surfaces_R = [concatenate([R_ie[::-1,i], R_oe[1:,i]]) for i in range(len(core_flux_grid))]
core_surfaces_z = [concatenate([z_ie[::-1,i], z_oe[1:,i]]) for i in range(len(core_flux_grid))]

outer_sol_surfaces_R = [concatenate([R_ol[::-1,len(pfr_flux_grid)+1+i], R_oe[1:,len(core_flux_grid)+1+i]]) for i in range(len(outer_sol_flux_grid)-1)]
outer_sol_surfaces_z = [concatenate([z_ol[::-1,len(pfr_flux_grid)+1+i], z_oe[1:,len(core_flux_grid)+1+i]]) for i in range(len(outer_sol_flux_grid)-1)]

inner_sol_surfaces_R = [concatenate([R_il[::-1,len(pfr_flux_grid)+1+i], R_ie[1:,len(core_flux_grid)+1+i]]) for i in range(len(inner_sol_flux_grid)-1)]
inner_sol_surfaces_z = [concatenate([z_il[::-1,len(pfr_flux_grid)+1+i], z_ie[1:,len(core_flux_grid)+1+i]]) for i in range(len(inner_sol_flux_grid)-1)]

outer_separatrix_R = concatenate([R_ol[::-1,len(pfr_flux_grid)], R_ie[1:,len(core_flux_grid)]])
outer_separatrix_z = concatenate([z_ol[::-1,len(pfr_flux_grid)], z_ie[1:,len(core_flux_grid)]])

inner_separatrix_R = concatenate([R_il[::-1,len(pfr_flux_grid)], R_oe[1:,len(core_flux_grid)]])
inner_separatrix_z = concatenate([z_il[::-1,len(pfr_flux_grid)], z_oe[1:,len(core_flux_grid)]])

[flux_surfaces_R.extend(S) for S in [pfr_surfaces_R, core_surfaces_R, outer_sol_surfaces_R, inner_sol_surfaces_R, [outer_separatrix_R, inner_separatrix_R]]]
[flux_surfaces_z.extend(S) for S in [pfr_surfaces_z, core_surfaces_z, outer_sol_surfaces_z, inner_sol_surfaces_z, [outer_separatrix_z, inner_separatrix_z]]]

















fig = plt.figure( figsize=(6,8))
ax1 = fig.add_subplot(111)

ax1.plot(*tcv_baffled_boundary(), lw = 2, c = 'black')

# for Rs, zs in zip(pfr_surfaces_R, pfr_surfaces_z):
#     ax1.plot(Rs, zs, c = 'purple')
#
# for Rs, zs in zip(core_surfaces_R, core_surfaces_z):
#     ax1.plot(Rs, zs, c = 'blue')
#
# for Rs, zs in zip(outer_sol_surfaces_R, outer_sol_surfaces_z):
#     ax1.plot(Rs, zs, c = 'green')
#
# for Rs, zs in zip(inner_sol_surfaces_R, inner_sol_surfaces_z):
#     ax1.plot(Rs, zs, c = 'orange')

for Rs, zs in zip(flux_surfaces_R, flux_surfaces_z):
    ax1.plot(Rs, zs)

# ax1.plot(outer_separatrix_R, outer_separatrix_z, c = 'red')
# ax1.plot(inner_separatrix_R, inner_separatrix_z, c = 'cyan')

ax1.axis('equal')
ax1.set_ylim([-0.8,-0.1])
ax1.set_xlim([0.6,1.1])
plt.tight_layout()
plt.show()







fig = plt.figure( figsize=(6,8) )
ax1 = fig.add_subplot(111)

ax1.plot(*tcv_baffled_boundary(), lw = 2, c = 'black')

for i in range(1,R_ol.shape[0]):
    ax1.plot(R_ol[i,:], z_ol[i,:], c = 'purple')

for i in range(1,R_il.shape[0]):
    ax1.plot(R_il[i,:], z_il[i,:], c = 'blue')

for i in range(1,R_oe.shape[0]):
    ax1.plot(R_oe[i,:], z_oe[i,:], c = 'orange')

for i in range(1,R_ie.shape[0]):
    ax1.plot(R_ie[i,:], z_ie[i,:], c = 'green')



# ax1.plot(outer_separatrix_R, outer_separatrix_z, c = 'red')
# ax1.plot(inner_separatrix_R, inner_separatrix_z, c = 'cyan')

ax1.axis('equal')
ax1.set_ylim([-0.8,-0.1])
ax1.set_xlim([0.6,1.1])
plt.tight_layout()
plt.show()





# check that every point in the meshes links correctly to a mesh vertex
bools = []
for R,z in zip(R_grids, z_grids):
    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            v = (R[i,j], z[i,j])
            bools.append(v in mesh.map)
print(f'# All grid points map correctly to a vertex? {all(bools)}')





para_laplace_inds = []
para_differences_inds = []
for R_surf,z_surf in zip(flux_surfaces_R, flux_surfaces_z):
    I = [mesh.map[(R,z)] for R,z in zip(R_surf, z_surf)]
    for k in range(len(I)-2):
        para_laplace_inds.append( (I[k], I[k+1], I[k+2]) )

    for k in range(len(I)-1):
        para_differences_inds.append( (I[k], I[k+1]) )


parallel_laplacian = zeros([len(para_laplace_inds), len(mesh.vertices)])
parallel_differences = zeros([len(para_differences_inds), len(mesh.vertices)])

for n, inds in enumerate(para_laplace_inds):
    i,j,k = inds
    parallel_laplacian[n,i] = 1
    parallel_laplacian[n,j] = -2
    parallel_laplacian[n,k] = 1

for n, inds in enumerate(para_differences_inds):
    i,j = inds
    parallel_differences[n,i] = -0.5
    parallel_differences[n,j] = 0.5







perp_laplace_inds = []
for i in range(1,R_ol.shape[0]):
    I = [mesh.map[(R, z)] for R, z in zip(R_ol[i,:], z_ol[i,:])]
    for k in range(len(I)-2):
        perp_laplace_inds.append( (I[k], I[k+1], I[k+2]) )

for i in range(1,R_il.shape[0]):
    I = [mesh.map[(R, z)] for R, z in zip(R_il[i,:], z_il[i,:])]
    for k in range(len(I)-2):
        perp_laplace_inds.append( (I[k], I[k+1], I[k+2]) )

for i in range(1,R_oe.shape[0]):
    I = [mesh.map[(R, z)] for R, z in zip(R_oe[i,:], z_oe[i,:])]
    for k in range(len(I)-2):
        perp_laplace_inds.append( (I[k], I[k+1], I[k+2]) )

for i in range(1,R_ie.shape[0]):
    I = [mesh.map[(R, z)] for R, z in zip(R_ie[i,:], z_ie[i,:])]
    for k in range(len(I)-2):
        perp_laplace_inds.append( (I[k], I[k+1], I[k+2]) )


perp_laplacian = zeros([len(perp_laplace_inds), len(mesh.vertices)])

for n, inds in enumerate(perp_laplace_inds):
    i,j,k = inds
    perp_laplacian[n,i] = 1
    perp_laplacian[n,j] = -2
    perp_laplacian[n,k] = 1




from mesh_tools.calcam_interface import save_geometry_matrix
from scipy.sparse import csc_matrix

parallel_laplacian = csc_matrix(parallel_laplacian)
parallel_laplacian.eliminate_zeros()
save_geometry_matrix(parallel_laplacian, filename = 'parallel_laplacian.npz')


perp_laplacian = csc_matrix(perp_laplacian)
perp_laplacian.eliminate_zeros()
save_geometry_matrix(perp_laplacian, filename = 'perp_laplacian.npz')







# def get_flux_surfaces(R_grids, z_grids, ):
#     flux_surfaces_R = []
#     flux_surfaces_z = []
#
#     pfr_surfaces_R = [concatenate([R_ol[::-1, i], R_il[1:, i]]) for i in range(len(pfr_flux_grid))]
#     pfr_surfaces_z = [concatenate([z_ol[::-1, i], z_il[1:, i]]) for i in range(len(pfr_flux_grid))]
#
#     core_surfaces_R = [concatenate([R_ie[::-1, i], R_oe[1:, i]]) for i in range(len(core_flux_grid))]
#     core_surfaces_z = [concatenate([z_ie[::-1, i], z_oe[1:, i]]) for i in range(len(core_flux_grid))]
#
#     outer_sol_surfaces_R = [concatenate([R_ol[::-1,len(pfr_flux_grid)+1+i], R_oe[1:,len(core_flux_grid)+1+i]]) for i in range(len(outer_sol_flux_grid)-1)]
#     outer_sol_surfaces_z = [concatenate([z_ol[::-1,len(pfr_flux_grid)+1+i], z_oe[1:,len(core_flux_grid)+1+i]]) for i in range(len(outer_sol_flux_grid)-1)]
#
#     inner_sol_surfaces_R = [concatenate([R_il[::-1,len(pfr_flux_grid)+1+i], R_ie[1:,len(core_flux_grid)+1+i]]) for i in range(len(inner_sol_flux_grid)-1)]
#     inner_sol_surfaces_z = [concatenate([z_il[::-1,len(pfr_flux_grid)+1+i], z_ie[1:,len(core_flux_grid)+1+i]]) for i in range(len(inner_sol_flux_grid)-1)]
#
#     outer_separatrix_R = concatenate([R_ol[::-1,len(pfr_flux_grid)], R_ie[1:,len(core_flux_grid)]])
#     outer_separatrix_z = concatenate([z_ol[::-1,len(pfr_flux_grid)], z_ie[1:,len(core_flux_grid)]])
#
#     inner_separatrix_R = concatenate([R_il[::-1,len(pfr_flux_grid)], R_oe[1:,len(core_flux_grid)]])
#     inner_separatrix_z = concatenate([z_il[::-1,len(pfr_flux_grid)], z_oe[1:,len(core_flux_grid)]])
#
#     [flux_surfaces_R.extend(S) for S in [pfr_surfaces_R, core_surfaces_R, outer_sol_surfaces_R, inner_sol_surfaces_R, [outer_separatrix_R, inner_separatrix_R]]]
#     [flux_surfaces_z.extend(S) for S in [pfr_surfaces_z, core_surfaces_z, outer_sol_surfaces_z, inner_sol_surfaces_z, [outer_separatrix_z, inner_separatrix_z]]]
#     return flux_surfaces_R, flux_surfaces_z
#
#
# def build_parallel_laplacian():
#     pass











print(parallel_laplacian.shape)
S = parallel_laplacian.T.dot(parallel_laplacian)
print(S.shape, len(mesh.vertices))

print( 'sparsity', S.count_nonzero() / S.shape[0]**2 )
print( 'sparsity', (parallel_laplacian == 0.).sum() / (parallel_laplacian.shape[0]*parallel_laplacian.shape[1]) )