
from numpy import array, dot, sqrt, sign
from scipy.interpolate import RectBivariateSpline


def dist(a,b):
    c = a - b
    return sqrt((c**2).sum())

def norm(a):
    return sqrt(dot(a,a))

class EqTracer(object):
    def __init__(self, R, z, psi):
        self.R = R
        self.z = z
        self.psi_grid = psi
        self.psi_spline = RectBivariateSpline(R,z,psi)

    def __call__(self,x, **kwargs):
        return self.psi_spline(*x, **kwargs).squeeze()

    def psi(self, x, **kwargs):
        return self.psi_spline(*x, **kwargs).squeeze()

    def grad(self,x):
        return array([self.psi(x,dx=1), self.psi(x,dy=1)])

    def grad_drn(self,x):
        g = array([self.psi(x,dx=1), self.psi(x,dy=1)])
        return g / sqrt(dot(g,g))

    def perp_grad_drn(self,x, direction = 1):
        p = array([self.psi(x,dy=1), -self.psi(x,dx=1)])
        return direction*p/sqrt(dot(p,p))

    def nabla(self,x):
        return self.psi(x,dx=1)**2 + self.psi(x,dy=1)**2

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
            mag = sqrt(dot(g,g))

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