
from numpy import array, zeros, concatenate, searchsorted, ndarray, ones, exp, diff
from scipy.special import factorial
from numpy.linalg import solve
from copy import deepcopy
import matplotlib.pyplot as plt
from morbo.boundary import Boundary

def cross_fade(G1, G2, sigma=0.05, k=2.):
    G3 = deepcopy(G1)
    x = zeros(G1.R.shape)
    for i in range(x.shape[1]):
        x[:,i] = G1.distance
    z = abs((x-1)/sigma)
    p = exp(-0.5*z**k)
    G3.R = G1.R*(1-p) + p*G2.R
    G3.z = G1.z*(1-p) + p*G2.z
    return G3

def get_fd_coeffs(points, order=1):
    # check validity of inputs
    if type(points) is not ndarray: points = array(points)
    n = len(points)
    if n <= order: raise ValueError('The order of the derivative must be less than the number of points')
    # build the linear system
    b = zeros(n)
    b[order] = factorial(order)
    A = ones([n,n])
    for i in range(1,n):
        A[i,:] = points**i
    # return the solution
    return solve(A,b)


class Grid(object):
    def __init__(self, flux_axis = None, parallel_axis = None):
        self.psi = flux_axis
        self.distance = parallel_axis
        self.R = zeros([len(parallel_axis), len(flux_axis)])
        self.z = zeros([len(parallel_axis), len(flux_axis)])
        self.lcfs_drn = zeros(2)
        self.lcfs_index = searchsorted(flux_axis, 1.0, side = 'left')
        self.trace_drn = 1
        self.condition = None

    def plot(self, ax = plt, color = 'black'):
        for i in range(self.R.shape[0]):
            ax.plot(self.R[i,:], self.z[i,:], c = color)
        for i in range(self.R.shape[1]):
            ax.plot(self.R[:,i], self.z[:,i], c = color)


class GridGenerator(object):
    def __init__(self, equilibrium = None, core_flux_grid = None, pfr_flux_grid = None, outer_sol_flux_grid = None,
                 inner_sol_flux_grid = None, inner_leg_distance_axis = None, outer_leg_distance_axis = None,
                 inner_edge_distance_axis = None, outer_edge_distance_axis = None, machine = None):

        self.eq = equilibrium
        """
        now use these to build the flux axes for the legs / edges
        """
        # TODO - parse inputs for validity

        inner_leg_psi_axis = concatenate([pfr_flux_grid,inner_sol_flux_grid])
        outer_leg_psi_axis = concatenate([pfr_flux_grid,outer_sol_flux_grid])
        inner_edge_psi_axis = concatenate([core_flux_grid,inner_sol_flux_grid])
        outer_edge_psi_axis = concatenate([core_flux_grid,outer_sol_flux_grid])

        """
        Build base grid objects
        """
        inner_leg = Grid(flux_axis = inner_leg_psi_axis, parallel_axis = inner_leg_distance_axis)
        outer_leg = Grid(flux_axis = outer_leg_psi_axis, parallel_axis = outer_leg_distance_axis)
        inner_edge = Grid(flux_axis = inner_edge_psi_axis, parallel_axis = inner_edge_distance_axis)
        outer_edge = Grid(flux_axis = outer_edge_psi_axis, parallel_axis = outer_edge_distance_axis)

        self.bound_poly = Boundary.load(machine)


        """
        Find the 4 directions pointing along the separatrix from the x-point
        """
        lcfs_directions = self.eq.lcfs_directions()
        il_drn, ie_drn, ol_drn, oe_drn = lcfs_directions
        inner_leg.lcfs_drn = il_drn
        outer_leg.lcfs_drn = ol_drn
        inner_edge.lcfs_drn = ie_drn
        outer_edge.lcfs_drn = oe_drn

        inner_leg.trace_drn = -1
        outer_leg.trace_drn = 1
        inner_edge.trace_drn = 1
        outer_edge.trace_drn = -1

        grids = [inner_leg, outer_leg, inner_edge, outer_edge]


        """
        form the grid boundaries
        """
        # create starting points for tracing the grid boundaries
        gap = 0.01
        for g in grids:
            start = self.eq.follow_gradient(self.eq.x_point + gap*g.lcfs_drn, target_psi=1.)
            # trace outer-leg boundary
            for i,p in enumerate(g.psi):
                x = self.eq.follow_gradient(start, target_psi = p)
                g.R[0,i] = x[0]
                g.z[0,i] = x[1]


        """
        match grids at x-point
        """
        for g in grids:
            g.R[0,g.lcfs_index] = self.eq.x_point[0]
            g.z[0,g.lcfs_index] = self.eq.x_point[1]


        """
        match the leg-edge boundaries
        """
        for G1, G2 in [(outer_leg, outer_edge), (inner_leg, inner_edge)]:
            boundary_R = 0.5*(G1.R[0,G1.lcfs_index:] + G2.R[0,G2.lcfs_index:])
            G1.R[0, G1.lcfs_index:] = boundary_R
            G2.R[0, G2.lcfs_index:] = boundary_R

            boundary_z = 0.5*(G1.z[0,G1.lcfs_index:] + G2.z[0,G2.lcfs_index:])
            G1.z[0, G1.lcfs_index:] = boundary_z
            G2.z[0, G2.lcfs_index:] = boundary_z



        """
        match the leg-leg and edge-edge boundaries
        """
        for G1, G2 in [(inner_leg, outer_leg), (inner_edge, outer_edge)]:
            boundary_R = 0.5*(G1.R[0,:G1.lcfs_index] + G2.R[0,:G2.lcfs_index])
            G1.R[0, :G1.lcfs_index] = boundary_R
            G2.R[0, :G2.lcfs_index] = boundary_R

            boundary_z = 0.5*(G1.z[0,:G1.lcfs_index] + G2.z[0,:G2.lcfs_index])
            G1.z[0, :G1.lcfs_index] = boundary_z
            G2.z[0, :G2.lcfs_index] = boundary_z


        inner_leg.condition = self.bound_poly.is_inside
        outer_leg.condition = self.bound_poly.is_inside
        inner_edge.condition = self.bound_poly.is_inside# lambda x : ~((x[1]>self.eq.magnetic_axis[1]) and (self.eq.grad(x)[0]>=0.))
        outer_edge.condition = self.bound_poly.is_inside# lambda x : ~((x[1]>self.eq.magnetic_axis[1]) and (self.eq.grad(x)[0]<=0.))



        self.inner_leg_dist_grid = deepcopy(inner_leg)
        self.outer_leg_dist_grid = deepcopy(outer_leg)
        self.inner_edge_dist_grid = deepcopy(inner_edge)
        self.outer_edge_dist_grid = deepcopy(outer_edge)

        self.inner_leg_ortho_grid = deepcopy(inner_leg)
        self.outer_leg_ortho_grid = deepcopy(outer_leg)
        self.inner_edge_ortho_grid = deepcopy(inner_edge)
        self.outer_edge_ortho_grid = deepcopy(outer_edge)

        self.distance_grids = [self.inner_leg_dist_grid, self.outer_leg_dist_grid,
                               self.inner_edge_dist_grid, self.outer_edge_dist_grid]
        self.orthogonal_grids = [self.inner_leg_ortho_grid, self.outer_leg_ortho_grid,
                                 self.inner_edge_ortho_grid, self.outer_edge_ortho_grid]

        for G in self.distance_grids:
            self.trace_distance_grid(G, step_size = 5e-3)

        for G in self.orthogonal_grids:
            self.trace_orthogonal_grid(G, step_size = 5e-3)

        self.inner_leg_grid = cross_fade(self.inner_leg_ortho_grid, self.inner_leg_dist_grid, sigma = 0.1)
        self.outer_leg_grid = cross_fade(self.outer_leg_ortho_grid, self.outer_leg_dist_grid, sigma = 0.01)
        self.inner_edge_grid = deepcopy(self.inner_edge_ortho_grid)
        self.outer_edge_grid = deepcopy(self.outer_edge_ortho_grid)

        self.grids = [self.inner_leg_grid, self.outer_leg_grid,
                      self.inner_edge_grid, self.outer_edge_grid]

    def plot_grids(self):
        cols = ['red', 'blue', 'green', 'violet']

        fig = plt.figure()
        ax1 = fig.add_subplot(131)
        for G,c in zip(self.distance_grids, cols):
            G.plot(color = c, ax = ax1)
        ax1.plot(self.bound_poly.x, self.bound_poly.y, c = 'black')
        ax1.axis('equal')

        ax2 = fig.add_subplot(132)
        for G,c in zip(self.orthogonal_grids, cols):
            G.plot(color = c, ax = ax2)
        ax2.plot(self.bound_poly.x, self.bound_poly.y, c = 'black')
        ax2.axis('equal')

        ax3 = fig.add_subplot(133)
        for G,c in zip(self.grids, cols):
            G.plot(color = c, ax = ax3)
        ax3.plot(self.bound_poly.x, self.bound_poly.y, c = 'black')
        ax3.axis('equal')
        plt.show()

    def trace_distance_grid(self, G, step_size = 1e-3):
        eps = 1e-3
        # First find the total poloidal distance to the endpoint along each flux surface
        total_poloidal_distance = zeros(len(G.psi))
        for i in range(len(G.psi)):
            if i != G.lcfs_index:
                v = array([G.R[0,i], G.z[0,i]])
                x0, distance = self.eq.follow_surface_while(v, G.condition, direction=G.trace_drn, step_size = step_size)
                total_poloidal_distance[i] = distance
            else:
                v = array([G.R[0,i] + eps*G.lcfs_drn[0], G.z[0,i] + eps*G.lcfs_drn[1]])
                x0, distance = self.eq.follow_surface_while(v, G.condition, direction=G.trace_drn, step_size = step_size)
                total_poloidal_distance[i] = distance + eps

            G.R[-1,i] = x0[0]
            G.z[-1,i] = x0[1]

        # now trace out point along each flux surface
        for j in range(len(G.psi)):
            gaps = diff(G.distance * total_poloidal_distance[j])

            v = array([G.R[0,j] + eps*G.lcfs_drn[0], G.z[0,j] + eps*G.lcfs_drn[1]])
            x0 = self.eq.follow_surface(v, gaps[0]-eps, direction = G.trace_drn, step_size = step_size)
            G.R[1,j] = x0[0]
            G.z[1,j] = x0[1]

            for i in range(1,len(G.distance)-2):
                v = array([G.R[i,j], G.z[i,j]])
                x0 = self.eq.follow_surface(v, gaps[i], direction=G.trace_drn, step_size = step_size)
                G.R[i+1,j] = x0[0]
                G.z[i+1,j] = x0[1]

    def trace_orthogonal_grid(self, G, step_size = 1e-3):
        """
        trace the grids using grad-psi
        """
        eps = 1e-3

        # First find the total poloidal distance to the endpoint along the separatrix
        v = array([G.R[0, G.lcfs_index] + eps*G.lcfs_drn[0], G.z[0, G.lcfs_index] + eps*G.lcfs_drn[1]])
        x0, lcfs_distance = self.eq.follow_surface_while(v, G.condition, direction=G.trace_drn, step_size = step_size)
        lcfs_distance += eps
        G.R[-1,G.lcfs_index] = x0[0]
        G.z[-1,G.lcfs_index] = x0[1]

        # Now trace out all the grid points along the separatrix
        gaps = diff(G.distance*lcfs_distance)
        x0 = self.eq.follow_surface(v, gaps[0]-eps, direction = G.trace_drn, step_size = step_size)
        G.R[1, G.lcfs_index] = x0[0]
        G.z[1, G.lcfs_index] = x0[1]

        for i,d in enumerate(gaps[1:]):
            v = array([G.R[i+1,G.lcfs_index], G.z[i+1,G.lcfs_index]])
            x0 = self.eq.follow_surface(v, d, direction=G.trace_drn, step_size = step_size)
            G.R[i+2,G.lcfs_index] = x0[0]
            G.z[i+2,G.lcfs_index] = x0[1]

        # Now trace all the orthogonal directions
        for i in range(1, len(G.distance)):
            # trace out the lower flux side
            for j in range(G.lcfs_index-1,-1,-1):
                v = array([G.R[i,j+1], G.z[i,j+1]])
                x0 = self.eq.follow_gradient(v, G.psi[j])
                G.R[i,j] = x0[0]
                G.z[i,j] = x0[1]
            # trace out the higher flux side
            for j in range(G.lcfs_index+1,len(G.psi)):
                v = array([G.R[i,j-1], G.z[i,j-1]])
                x0 = self.eq.follow_gradient(v, G.psi[j])
                G.R[i,j] = x0[0]
                G.z[i,j] = x0[1]


    def other_memes(self):
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