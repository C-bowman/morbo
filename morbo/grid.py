
from numpy import array, zeros, linspace, concatenate, searchsorted
from mesh_tools.vessel_boundaries import tcv_baffled_boundary
from copy import deepcopy
import matplotlib.pyplot as plt


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
    def __init__(self, equilibrium = None, core_flux_grid = None, pfr_flux_grid = None,
                 outer_sol_flux_grid = None, inner_sol_flux_grid = None):

        self.eq = equilibrium
        """
        now use these to build the flux axes for the legs / edges
        """
        outer_leg_psi_axis = []
        outer_leg_psi_axis.extend(pfr_flux_grid)
        outer_leg_psi_axis.append(1.)
        outer_leg_psi_axis.extend(outer_sol_flux_grid)

        outer_edge_psi_axis = []
        outer_edge_psi_axis.extend(core_flux_grid)
        outer_edge_psi_axis.append(1.)
        outer_edge_psi_axis.extend(outer_sol_flux_grid)

        inner_edge_psi_axis = []
        inner_edge_psi_axis.extend(core_flux_grid)
        inner_edge_psi_axis.append(1.)
        inner_edge_psi_axis.extend(inner_sol_flux_grid)

        inner_leg_psi_axis = []
        inner_leg_psi_axis.extend(pfr_flux_grid)
        inner_leg_psi_axis.append(1.)
        inner_leg_psi_axis.extend(inner_sol_flux_grid)


        """
        specify the poloidal distance grids
        """
        outer_leg_distance_axis = concatenate([linspace(0,0.9,16), linspace(0.9,1,6)[1:]])
        inner_leg_distance_axis = concatenate([linspace(0,0.65,5), linspace(0.65,1,6)[1:]])

        outer_edge_distance_axis = linspace(0,1,30)
        inner_edge_distance_axis = linspace(0,1,30)


        inner_leg = Grid(flux_axis = inner_leg_psi_axis, parallel_axis = inner_leg_distance_axis)
        outer_leg = Grid(flux_axis = outer_leg_psi_axis, parallel_axis = outer_leg_distance_axis)
        inner_edge = Grid(flux_axis = inner_edge_psi_axis, parallel_axis = inner_edge_distance_axis)
        outer_edge = Grid(flux_axis = outer_edge_psi_axis, parallel_axis = outer_edge_distance_axis)


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


        from mesh_tools.mesh import Polygon
        bound_poly = Polygon(*tcv_baffled_boundary())

        inner_leg.condition = bound_poly.is_inside
        outer_leg.condition = bound_poly.is_inside
        inner_edge.condition = lambda x : ~((x[1]>self.eq.magnetic_axis[1]) and (self.eq.grad(x)[0]>=0.))
        outer_edge.condition = lambda x : ~((x[1]>self.eq.magnetic_axis[1]) and (self.eq.grad(x)[0]<=0.))

        inner_leg_orth = deepcopy(inner_leg)
        outer_leg_orth = deepcopy(outer_leg)
        inner_edge_orth = deepcopy(inner_edge)
        outer_edge_orth = deepcopy(outer_edge)
        orth_grids = [inner_leg_orth, outer_leg_orth, inner_edge_orth, outer_edge_orth]

        for G in grids:
            self.trace_distance_grid(G)

        for G in orth_grids:
            self.trace_orthogonal_grid(G)


        inner_leg.plot(color = 'red')
        outer_leg.plot(color = 'blue')
        inner_edge.plot(color = 'green')
        outer_edge.plot(color = 'violet')
        plt.plot(*tcv_baffled_boundary(), c = 'black')
        plt.axis('equal')
        plt.show()

        inner_leg_orth.plot(color = 'red')
        outer_leg_orth.plot(color = 'blue')
        inner_edge_orth.plot(color = 'green')
        outer_edge_orth.plot(color = 'violet')
        plt.plot(*tcv_baffled_boundary(), c = 'black')
        plt.axis('equal')
        plt.show()



    def trace_distance_grid(self, G):
        total_poloidal_distance = zeros(len(G.psi))
        for i in range(len(G.psi)):
            if i != G.lcfs_index:
                v = array([G.R[0,i], G.z[0,i]])
                x0, distance = self.eq.follow_surface_while(v, G.condition, direction=G.trace_drn)
                total_poloidal_distance[i] = distance
            else:
                v = array([G.R[0,i] + 0.001*G.lcfs_drn[0], G.z[0,i] + 0.001*G.lcfs_drn[1]])
                x0, distance = self.eq.follow_surface_while(v, G.condition, direction=G.trace_drn)
                total_poloidal_distance[i] = distance + 0.001

            G.R[-1,i] = x0[0]
            G.z[-1,i] = x0[1]

        for j in range(len(G.psi)):
            gaps = G.distance * total_poloidal_distance[j]
            for i,d in enumerate(gaps[1:-1]):
                if j != G.lcfs_index and i != 0:
                    v = array([G.R[0,j], G.z[0,j]])
                    x0 = self.eq.follow_surface(v, d, direction=G.trace_drn)
                else:
                    v = array([G.R[0,j] + 0.001*G.lcfs_drn[0], G.z[0,j] + 0.001*G.lcfs_drn[1]])
                    x0 = self.eq.follow_surface(v, d, direction=G.trace_drn)

                G.R[i+1,j] = x0[0]
                G.z[i+1,j] = x0[1]


    def trace_orthogonal_grid(self, G):
        """
        trace the grids using grad-psi
        """
        total_poloidal_distance = zeros(len(G.psi))
        for i in range(len(G.psi)):
            if i != G.lcfs_index:
                v = array([G.R[0,i], G.z[0,i]])
                x0, distance = self.eq.follow_surface_while(v, G.condition, direction=G.trace_drn)
                total_poloidal_distance[i] = distance
            else:
                v = array([G.R[0,i] + 0.001*G.lcfs_drn[0], G.z[0,i] + 0.001*G.lcfs_drn[1]])
                x0, distance = self.eq.follow_surface_while(v, G.condition, direction=G.trace_drn)
                total_poloidal_distance[i] = distance + 0.001

            G.R[-1,i] = x0[0]
            G.z[-1,i] = x0[1]

        # now trace all the points in the separatrix
        gaps = G.distance * total_poloidal_distance[G.lcfs_index]
        for i,d in enumerate(gaps[1:]):
            v = array([G.R[0,G.lcfs_index] + 0.001*G.lcfs_drn[0], G.z[0,G.lcfs_index] + 0.001*G.lcfs_drn[1]])
            x0 = self.eq.follow_surface(v, d, direction=G.trace_drn)

            G.R[i+1,G.lcfs_index] = x0[0]
            G.z[i+1,G.lcfs_index] = x0[1]

        # now
        for j in range(len(G.psi)):
            gaps = G.distance * total_poloidal_distance[j]
            for i,d in enumerate(gaps[1:]):
                if j != G.lcfs_index:
                    v = array([G.R[i+1,G.lcfs_index], G.z[i+1,G.lcfs_index]])
                    x0 = self.eq.follow_gradient(v, G.psi[j])
                    G.R[i+1,j] = x0[0]
                    G.z[i+1,j] = x0[1]


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