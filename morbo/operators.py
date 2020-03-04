
from numpy import array, zeros, ndarray, ones, sqrt
from scipy.special import factorial
from numpy.linalg import solve


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


def parallel_derivative(grid, order = 2):
    info = {}
    for i in range(1,len(grid.distance)-1):
        for j in range(len(grid.psi)):
            p0 = (grid.R[i-1,j], grid.z[i-1,j])
            p1 = (grid.R[i,j], grid.z[i,j])
            p2 = (grid.R[i+1,j], grid.z[i+1,j])

            d01 = sqrt((p0[0]-p1[0])**2 + (p0[1]-p1[1])**2)
            d12 = sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
            pol_dist = array([0., d01, d01+d12])
            coeffs = get_fd_coeffs(pol_dist, order=order)

            info[p1] = {p0 : coeffs[0],
                        p1 : coeffs[1],
                        p2 : coeffs[2]}
    return info


def perpendicular_derivative(grid, order = 2, use_flux = True):
    info = {}
    for i in range(len(grid.distance)):
        for j in range(1,len(grid.psi)-1):
            p0 = (grid.R[i,j-1], grid.z[i,j-1])
            p1 = (grid.R[i,j], grid.z[i,j])
            p2 = (grid.R[i,j+1], grid.z[i,j+1])

            if use_flux:
                dist = grid.psi[j-1:j+2]
            else:
                d01 = sqrt((p0[0]-p1[0])**2 + (p0[1]-p1[1])**2)
                d12 = sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
                dist = array([0., d01, d01+d12])

            coeffs = get_fd_coeffs(dist, order=order)
            info[p1] = {p0 : coeffs[0],
                        p1 : coeffs[1],
                        p2 : coeffs[2]}
    return info