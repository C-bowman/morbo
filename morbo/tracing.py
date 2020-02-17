
from numpy import array, dot, sqrt, sign, linspace, meshgrid, pi, sin, cos
from scipy.interpolate import RectBivariateSpline
from itertools import product
import matplotlib.pyplot as plt


def dist(a,b):
    c = a - b
    return sqrt((c**2).sum())

def norm(a):
    return sqrt(dot(a,a))

def unit(a):
    return a/sqrt(dot(a,a))



class Equilibrium(object):
    def __init__(self, R, z, psi):
        self.R = R
        self.z = z
        self.psi_grid = psi
        self.psi_spline = RectBivariateSpline(R,z,psi)

        self.R_max = R.max()
        self.R_min = R.min()
        self.z_max = z.max()
        self.z_min = z.min()

        self.normalise_flux()

    def __call__(self,x, **kwargs):
        return self.psi_spline.ev(*x, **kwargs).squeeze()

    def psi(self, x, **kwargs):
        return self.psi_spline.ev(*x, **kwargs).squeeze()

    def grad(self,x):
        return array([self.psi(x,dx=1), self.psi(x,dy=1)])

    def grad_drn(self,x):
        g = array([self.psi(x,dx=1), self.psi(x,dy=1)])
        return unit(g)

    def perp_grad_drn(self, x, direction = 1):
        p = array([self.psi(x,dy=1), -self.psi(x,dx=1)])
        return direction*unit(p)

    def nabla(self,x):
        return self.psi(x,dx=1)**2 + self.psi(x,dy=1)**2

    def grad_nabla(self,x):
        return 2*self.hessian(x).dot(self.grad(x))

    def hessian(self,x):
        H = array([[self.psi(x,dx=2), self.psi(x,dx=1, dy=1)],
                   [self.psi(x,dx=1, dy=1), self.psi(x,dy=2)]])
        return H

    def newton_update(self,x,target):
        g = self.grad(x)
        return (target - self.psi(x)) * g / dot(g,g)

    def follow_gradient(self, x0, target_psi, max_step_size = 5e-3):
        psi0 = self.psi(x0)
        drn = sign(target_psi - psi0)
        d = abs(target_psi - psi0)

        # step along the gradient until we overshoot the target
        while drn*(target_psi-self.psi(x0)) > 0:
            g = self.grad(x0)
            mag = norm(g)

            s = drn*min(0.05*d/mag, max_step_size)
            x0 += s*g/mag

        # now aim for the target
        for i in range(2):
            x0 += self.newton_update(x0, target=target_psi)

        return x0

    def follow_surface(self, start, distance, step_size = 1e-3, max_steps = 2000, direction = 1):
        """
        Follows the current flux surface until a given poloidal distance has been travelled.
        """
        x0 = start.copy()
        psi0 = self.psi(x0)
        travelled = 0

        for i in range(max_steps):
            # take a step along the surface
            x1 = x0 + step_size*self.perp_grad_drn(x0, direction = direction)
            # apply one newton-method update to keep us on the flux surface
            x1 += self.newton_update(x1, target=psi0)
            # check the distance travelled
            d = dist(x0,x1)
            remaining_distance = distance-travelled
            if d < remaining_distance: # if we need to go further, accept the step
                travelled += d
                x0 = x1.copy()
            else: # if we've overshot, reject the step and cut the step size
                step_size = 0.95*remaining_distance
            # break the loop once we're acceptably close
            if remaining_distance < 1e-6: break
        return x0

    def follow_surface_while(self, start, condition, step_size = 1e-3, max_steps = 2000, direction = 1):
        """
        Follows the current flux surface while the current position meets a provided condition
        """
        x0 = start.copy()
        psi0 = self.psi(x0)
        travelled = 0

        for i in range(max_steps):
            # take a step along the surface
            x1 = x0 + step_size*self.perp_grad_drn(x0, direction = direction)
            # apply one newton-method update to keep us on the flux surface
            x1 += self.newton_update(x1, target=psi0)
            # check the if the condition is still True
            if condition(x1): # if True, accept the step and keep following the surface
                travelled += dist(x0,x1)
                x0 = x1.copy()
            else: # if False, cut the step size in half to binary-search for the boundary
                step_size *= 0.5
            # break the loop once the step size is small enough to ensure good accuracy
            if step_size < 1e-6: break

        return x0, travelled

    def find_stationary_points(self, R_points = 3, z_points = 25):
        """
        Find all points inside the flux grid where the magnitude of the gradient is zero
        """
        # create a grid of locations to run root finding from
        z_starts = linspace(self.z_min, self.z_max, z_points+2)[1:-1]
        R_starts = linspace(self.R_min, self.R_max, R_points+2)[1:-1]
        alpha = 1. # step size multiplier on the root-finding update

        minima = []
        for R0,z0 in product(R_starts, z_starts):
            coords = [array([R0, z0])]

            for i in range(50):
                p = self.nabla(coords[-1])
                g = self.grad_nabla(coords[-1])
                update = -alpha*p*g/dot(g,g)
                new = coords[-1] + update
                coords.append(new)
                # terminate the search if we've left the bounds
                if not ((self.R_min < new[0] < self.R_max) and (self.z_min < new[1] < self.z_max)): break
                # if the update size
                if norm(update) < 1e-8:
                    minima.append(new)
                    break

        # filter out the duplicate points
        stationary_points = [minima.pop()]
        for p in minima:
            if not any([dist(p,s)<1e-4 for s in stationary_points]):
                stationary_points.append(p)

        return stationary_points

    def normalise_flux(self):
        # get all stationary points
        points = self.find_stationary_points()
        # sort them by their flux value
        points = sorted(points, key = self.psi)
        # assign the axis and x-point
        self.magnetic_axis, self.x_point, *_ = points
        # normalise the flux and re-build the spline
        self.psi_grid = (self.psi_grid - self.psi(self.magnetic_axis)) / (self.psi(self.x_point)-self.psi(self.magnetic_axis))
        self.psi_spline = RectBivariateSpline(self.R, self.z, self.psi_grid)

    def lcfs_directions(self):
        """
        Find the 4 unit vectors which point along the direction of the separatrix at the x-point
        """
        # brute-force grid-search to find a direction pointing along the LCFS
        theta = linspace(0, 2*pi, 4*360)
        r = 0.01
        lcfs_deviation = (1 - self.psi([self.x_point[0] + r*cos(theta), self.x_point[1] + r*sin(theta)])) ** 2
        lcfs_angle = theta[lcfs_deviation.argmin()]
        # now get the other 3 LCFS directions
        cardinals = [0, 0.5*pi, pi, 1.5*pi]
        lcfs_directions = [array([cos(lcfs_angle + t), sin(lcfs_angle + t)]) for t in cardinals]

        # get the direction vector from the lower x-point to the axis
        axis_drn = unit(array([self.magnetic_axis[0]-self.x_point[0], self.magnetic_axis[1]-self.x_point[1]]))
        # also get the outboard direction
        outboard_drn = array([1,0])
        # first sort by leg / edge region
        lcfs_directions = sorted(lcfs_directions, key = lambda x:sign(dot(x, axis_drn)))
        # then sort by inner / outer
        lcfs_directions = sorted(lcfs_directions, key = lambda x:sign(dot(x, outboard_drn)))
        # this means the order should be inner leg, inner edge, outer leg, outer edge
        return lcfs_directions

    def plot_equilibrium(self):
        R_mesh, z_mesh = meshgrid(linspace(self.R_min, self.R_max, 64), linspace(self.z_min, self.z_max, 128))

        aspect = (self.R_max-self.R_min)/(self.z_max-self.z_min)
        plt.figure(figsize = (10.*aspect,10.))
        psi_mesh = self.psi([R_mesh, z_mesh])
        plt.contourf(R_mesh, z_mesh, psi_mesh, 32)
        plt.contour(R_mesh, z_mesh, psi_mesh, levels = [1.], colors = ['red'])
        plt.plot(*self.magnetic_axis, 'x', color = 'dodgerblue', label = 'magnetic axis', markersize = 8)
        # plt.plot(*self.x_point, 'x', color = 'red', label = 'X-point')
        plt.xlim([self.R_min,self.R_max])
        plt.ylim([self.z_min,self.z_max])
        plt.legend()
        plt.axis('equal')
        plt.show()

    def plot_stationary_points(self):
        R_mesh, z_mesh = meshgrid(linspace(self.R_min, self.R_max, 64), linspace(self.z_min, self.z_max, 128))

        points = self.find_stationary_points()

        aspect = (self.R_max-self.R_min)/(self.z_max-self.z_min)
        plt.figure(figsize = (10.*aspect,10.))
        psi_mesh = self.psi([R_mesh, z_mesh])
        plt.contour(R_mesh, z_mesh, psi_mesh, 32)
        for p in points:
            plt.plot(*p, 'x', color = 'red')
        plt.xlim([self.R_min,self.R_max])
        plt.ylim([self.z_min,self.z_max])
        plt.legend()
        plt.axis('equal')
        plt.show()

