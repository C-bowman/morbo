
from numpy import array, append, where, linspace, zeros, sqrt


class Boundary(object):
    def __init__(self, x, y):
        self.x = array(x)
        self.y = array(y)
        self.n = len(x)

        self.im = []
        self.c = []
        self.x_upr = []
        self.x_lwr = []
        self.y_upr = []
        self.y_lwr = []
        self.dx = []
        self.dy = []

        for i in range(self.n-1):
            dx = self.x[i+1] - self.x[i]
            dy = self.y[i+1] - self.y[i]

            self.dx.append(dx)
            self.dy.append(dy)

            self.im.append(dx/dy)
            self.c.append(self.y[i] - self.x[i]*(dy/dx))

            self.x_upr.append(max(self.x[i+1], self.x[i]))
            self.x_lwr.append(min(self.x[i+1], self.x[i]))

            self.y_upr.append(max(self.y[i+1], self.y[i]))
            self.y_lwr.append(min(self.y[i+1], self.y[i]))


        self.im = array(self.im)
        self.c = array(self.c)
        self.x_lwr = array(self.x_lwr)
        self.x_upr = array(self.x_upr)
        self.y_lwr = array(self.y_lwr)
        self.y_upr = array(self.y_upr)
        self.dx = array(self.dx)
        self.dy = array(self.dy)

        # normalise the unit vectors
        self.lengths = sqrt(self.dx**2 + self.dy**2)
        self.dx /= self.lengths
        self.dy /= self.lengths

        self.zero_im = self.im == 0.

    @classmethod
    def load(cls, machine):
        machine_boundaries = {
            'mast_u' : mastu_boundary,
            'tcv' : tcv_baffled_boundary,
            'jet' : jet_boundary_detailed
        }
        return cls(*machine_boundaries[machine]())

    def is_inside(self, v):
        x, y = v
        k = (y - self.c)*self.im

        # limits_check = (self.y_lwr < y) & (y < self.y_upr) & (x < self.x_upr)
        # isec_check = ( (self.x_lwr < k) & (k < self.x_upr) & (x < k)) | self.zero_im

        limits_check = (self.y_lwr <= y) & (y <= self.y_upr) & (x <= self.x_upr)
        isec_check = ( (self.x_lwr <= k) & (k <= self.x_upr) & (x <= k)) | self.zero_im

        intersections = sum(limits_check & isec_check)
        if intersections % 2 == 0:
            return False
        else:
            return True

    def distance(self, v):
        x, y = v

        dx = x - self.x[:-1]
        dy = y - self.y[:-1]

        L = (dx*self.dx + dy*self.dy) / self.lengths
        D = dx*self.dy - dy*self.dx
        booles = (0 <= L) & (L <= 1)

        points_min = sqrt(dx**2 + dy**2).min()

        if any(booles):
            perp_min = abs(D[where(booles)]).min()
            return min(perp_min, points_min)
        else:
            return points_min

    def diagnostic_plot(self):

        xmin = self.x.min()
        xmax = self.x.max()
        ymin = self.y.min()
        ymax = self.y.max()
        xpad = (xmax-xmin)*0.15
        ypad = (ymax-ymin)*0.15

        N = 200
        x_ax = linspace(xmin-xpad, xmax+xpad, N)
        y_ax = linspace(ymin-ypad, ymax+ypad, N)

        inside = zeros([N,N])
        distance = zeros([N,N])
        for i in range(N):
            for j in range(N):
                v = [x_ax[i], y_ax[j]]
                inside[i,j] = self.is_inside(v)
                distance[i,j] = self.distance(v)

        import matplotlib.pyplot as plt
        plt.contourf(x_ax, y_ax, inside.T)
        plt.plot(self.x, self.y, '.-', c = 'white', lw = 2)
        plt.show()

        plt.contourf(x_ax, y_ax, distance.T, 100)
        plt.plot(self.x, self.y, '.-', c='white', lw=2)
        plt.show()

        plt.contourf(x_ax, y_ax, (distance*inside).T, 100)
        plt.plot(self.x, self.y, '.-', c='white', lw=2)
        plt.show()




def mastu_divertor_geometry():

    r = [1250.27, 1348.3, 1470, 1470, 1450, 1450, 1321.37, 1190.43,
         892.96, 869.38, 839.81, 822.29, 819.74, 819.74, 827.34,
         854.8, 890.17, 919.74, 940.66, 1555, 1850, 2000,
         2000, 2000, 2000, 1318.8, 1768.9, 1730.071, 1350,
         1090, 1090, 905.756, 878.886, 878.886, 907.17, 539.48,
         511.196, 505.534, 535.94, 507.4, 497.4, 507.4, 478.8,
         468.8, 478.8, 333, 333, 275, 334, 261, 261, 244, 261, 261,
         244, 261, 261]

    z = [1000, 860, 860, 810, 810, 820, 820, 1007, 1303.96, 1331.21,
         1382.58, 1445.1, 1481.24, 1493.6, 1531.85, 1569.64, 1589.13,
         1593.6, 1593.58, 1567.0, 1080.0, 1080.0, 1700.0, 2035, 2169,
         2169, 1718.9, 1680, 2060, 2060, 2060, 1878.584, 1905.454,
         1905.454, 1877.17, 1509.48, 1537.764, 1532.106, 1501.70,
         1473.756, 1483.756, 1473.756, 1445.754, 1455.754, 1445.754,
         1303, 1100, 1100, 1100, 502, 348, 348, 348, 146, 146, 146, 0]

    r = array(r) * 1e3
    z = array(z) * 1e3

    r = append(r, r[::-1])
    z = append(z, -z[::-1])

    return r, z




def mastu_boundary():
    r = [2.000, 2.000, 1.191, 0.893, 0.868, 0.847, 0.832, 0.823, 0.820,
         0.820, 0.825, 0.840, 0.864, 0.893, 0.925, 1.690, 2.000, 2.000,
         1.319, 1.769, 1.730, 1.350, 1.090, 0.900, 0.360, 0.333, 0.333,
         0.261, 0.261, 0.261, 0.333, 0.333, 0.360, 0.900, 1.090, 1.350,
         1.730, 1.769, 1.319, 2.000, 2.000, 1.690, 0.925, 0.893, 0.864,
         0.840, 0.825, 0.820, 0.820, 0.823, 0.832, 0.847, 0.868, 0.893,
         1.191, 2.000, 2.000]

    z = [-0.000, -1.007, -1.007, -1.304, -1.334, -1.368, -1.404, -1.442,
         -1.481, -1.490, -1.522, -1.551, -1.573, -1.587, -1.590, -1.552,
         -1.560, -2.169, -2.169, -1.719, -1.680, -2.060, -2.060, -1.870,
         -1.330, -1.303, -1.100, -0.500,  0.000,  0.500,  1.100,  1.303,
          1.330,  1.870,  2.060,  2.060,  1.680,  1.719,  2.169,  2.169,
          1.560,  1.552,  1.590,  1.587,  1.573,  1.551,  1.522,  1.490,
          1.481,  1.442,  1.404,  1.368,  1.334,  1.304,  1.007,  1.007,
          0.000]

    r = array(r); z = array(z)
    return r, z




def superx_boundary():
    r = [ 0.552,  0.825,  0.840,  0.864,  0.893,  0.925,  1.690,
          1.730,  1.730,  1.350,  1.090,  0.900,  0.552]
    z = [-1.522, -1.522, -1.551, -1.573, -1.587, -1.590, -1.552,
         -1.552, -1.680, -2.060, -2.060, -1.870, -1.522]
    r = array(r); z = array(z)
    return r, z




def superx_T5_grid_boundary():
    r = [ 0.617,  0.893,  1.400,  1.694,
          1.694,  1.350,  1.090,  0.900,  0.617]
    z = [-1.587, -1.587, -1.669, -1.669,
         -1.716, -2.060, -2.060, -1.870, -1.587]
    r = array(r); z = array(z)
    return r, z




def jet_boundary(z_limit = None):
    # Define the JET wall geometry
    wall_r = [2.01001,    2.08113003, 2.13895011, 2.19957995, 2.29543996, 2.35993004,
              2.40478992, 2.41290998, 2.41257,    2.40761995, 2.39801002, 2.4210701,
              2.41774011, 2.40573001, 2.36837006, 2.31498003, 2.36430001, 2.43665004,
              2.52368999, 2.57577991, 2.81418991, 2.81418991, 2.8701601,  2.94647002,
              2.98697996, 2.93504,    2.89768004, 2.88123012, 2.88391995, 2.90044999,
              2.89049006, 2.88591003, 2.88753009, 2.9066,     2.96348,    3.01095009,
              3.06005001, 3.19404006, 3.30938005, 3.35908008, 3.4073,     3.47948003,
              3.53044009, 3.57817006, 3.62704992, 3.64053988, 3.68002009, 3.71647,
              3.74942994, 3.77921009, 3.80541992, 3.82831001, 3.84758997, 3.86342001,
              3.87561989, 3.88427997, 3.88930988, 3.89074993, 3.88857007, 3.88276005,
              3.87339997, 3.86038995, 3.84390998, 3.82380009, 3.8003099,  3.77324009,
              3.74291992, 3.70907998, 3.6721499,  3.64092994, 3.64041996, 3.56203008,
              3.47712994, 3.39778996, 3.3001399,  3.1494,     2.97530007, 2.88351989,
              2.7846899,  2.67780995, 2.57509995, 2.47879004, 2.38113999, 2.24335003,
              2.20596004, 2.17813993, 2.11916995, 2.05840993, 1.99741995, 1.93823004,
              1.91601002, 1.90404999, 1.88606,    1.87135994, 1.85827005, 1.84517002,
              1.83206999, 1.82012999, 1.81194997, 1.80708003, 1.80551004, 1.80726004,
              1.81233001, 1.82069004, 1.83236003, 1.84730005, 1.86552,    1.88698006,
              1.91165996, 1.95395994, 2.01001,    2.01001]

    wall_z = [-0.78684998, -0.95784998, -1.09745002, -1.24389005, -1.33442998, -1.33442998,
              -1.38585997, -1.42893004, -1.47444999, -1.50441003, -1.51640999, -1.60172999,
              -1.64941001, -1.68971002, -1.68971002, -1.73870003, -1.73792005, -1.71074998,
              -1.70983005, -1.59885000, -1.66859000, -1.70844996, -1.71311998, -1.74476004,
              -1.74594998, -1.68233001, -1.68233001, -1.61945999, -1.58169997, -1.51040995,
              -1.49840999, -1.47573996, -1.42303002, -1.38336003, -1.33481002, -1.33481002,
              -1.29777002, -1.21404004, -1.07955003, -1.03118002, -0.97816998, -0.89091998,
              -0.82266003, -0.75269997, -0.67401999, -0.65024,    -0.57538003, -0.49847999,
              -0.42052001, -0.34079999, -0.26032999, -0.17837,    -0.09595,    -0.01234,
               0.07141,     0.15606999,  0.24056,     0.32565001,  0.41025001,  0.49515,
               0.57927001,  0.66337001,  0.74637997,  0.82906997,  0.91039002,  0.99106997,
               1.07008004,  1.14816999,  1.22432995,  1.28314996,  1.42449999,  1.49509001,
               1.57154,     1.64297998,  1.71131003,  1.81630003,  1.94143999,  1.97645998,
               2.00469995,  2.0183301,   2.0157001,   1.99909997,  1.96721995,  1.89377999,
               1.87144995,  1.83916998,  1.72309005,  1.60283005,  1.44400001,  1.28935003,
               1.23155999,  1.17127001,  1.07288003,  0.97373003,  0.87440002,  0.77499998,
               0.67563999,  0.57594001,  0.47635999,  0.37656,     0.27666,     0.17676,
               0.07698,    -0.02259,    -0.12182,    -0.22060999, -0.31885001, -0.41643,
              -0.51324999, -0.65587997, -0.78684998, -0.78684998]

    wall_z = array(wall_z)
    wall_r = array(wall_r)

    if z_limit is not None:
        inds = where(wall_z <= z_limit)
        wall_z = wall_z[inds]
        wall_r = wall_r[inds]

        wall_z = append(wall_z, wall_z[0])
        wall_r = append(wall_r, wall_r[0])

    return wall_r, wall_z





def jet_boundary_detailed(z_limit = None):
    # Define the JET wall geometry

    wall_r =[2.1446, 2.2936, 2.2936, 2.2954, 2.3599, 2.3961, 2.4019, 2.4047, 2.4074, 2.409,
             2.4121, 2.4128, 2.4129, 2.4127, 2.4124, 2.4121, 2.4075, 2.3979, 2.4191, 2.4203,
             2.4209, 2.4212, 2.4212, 2.4211, 2.4187, 2.4185, 2.4179, 2.417 , 2.4162, 2.4056,
             2.3149, 2.3534, 2.3593, 2.3631, 2.3669, 2.3706, 2.3742, 2.4273, 2.431 , 2.4347,
             2.4385, 2.4423, 2.4461, 2.5236, 2.5245, 2.5236, 2.5226, 2.5223, 2.523 , 2.5588,
             2.5604, 2.5622, 2.5526, 2.5516, 2.5518, 2.5738, 2.6325, 2.6336, 2.6934, 2.694,
             2.754 , 2.7549, 2.8143, 2.8143, 2.8041, 2.8568, 2.8614, 2.8651, 2.8688, 2.8724,
             2.876 , 2.8783, 2.9362, 2.9406, 2.9442, 2.9478, 2.9515, 2.9534, 2.9571, 2.9868,
             2.8975, 2.8818, 2.881 , 2.8803, 2.8801, 2.8801, 2.8805, 2.8814, 2.9002, 2.8903,
             2.8876, 2.8868, 2.8862, 2.8858, 2.8857, 2.8857, 2.8859, 2.8863, 2.8871, 2.8893,
             2.9006, 2.9023, 2.9053, 2.9075, 2.9132, 2.9633, 3.0095, 3.0107, 3.0132, 3.0214,
             3.0967, 3.0598, 3.1938, 3.1959, 3.1973, 3.1988, 3.202 , 3.306 , 3.306 , 3.2897,
             3.2778, 3.2739, 3.2728, 3.2721, 3.2721, 3.2724, 3.2738, 3.2828, 3.3116, 3.3195,
             3.3281, 3.3521, 3.3544, 3.3593, 3.3617, 3.3643, 3.3728, 3.423 , 3.4298, 3.4346,
             3.4369, 3.4392, 3.4469, 3.494 , 3.5004, 3.505 , 3.5071, 3.5091, 3.5165, 3.5588,
             3.5658, 3.5703, 3.5718, 3.5735, 3.5802, 3.6189, 3.6254, 3.6297, 3.6314, 3.6326,
             3.6387, 3.6732, 3.6788, 3.6828, 3.6846, 3.6855, 3.6876, 3.7219, 3.7266, 3.7304,
             3.732 , 3.7327, 3.7346, 3.7678, 3.7685, 3.772 , 3.7736, 3.7739, 3.7754, 3.8022,
             3.8044, 3.8077, 3.8091, 3.8091, 3.8104, 3.8329, 3.8342, 3.8372, 3.8385, 3.8381,
             3.8392, 3.8561, 3.8577, 3.8604, 3.8616, 3.8609, 3.8616, 3.8748, 3.8748, 3.8773,
             3.8784, 3.8774, 3.8779, 3.8852, 3.8857, 3.8879, 3.8888, 3.888 , 3.8875, 3.89,
             3.8902, 3.892 , 3.8928, 3.8912, 3.891 , 3.8882, 3.8882, 3.8897, 3.8904, 3.888,
             3.888 , 3.8799, 3.8799, 3.8809, 3.8815, 3.8793, 3.8785, 3.8651, 3.8651, 3.8658,
             3.8663, 3.8637, 3.8631, 3.8441, 3.844 , 3.8444, 3.8448, 3.8419, 3.8405, 3.8167,
             3.8167, 3.8167, 3.8138, 3.8126, 3.7833, 3.7828, 3.7827, 3.7827, 3.7795, 3.7786,
             3.7437, 3.7432, 3.7428, 3.743 , 3.7391, 3.7378, 3.6983, 3.6977, 3.6969, 3.6967,
             3.6928, 3.6914, 3.6745, 3.6696, 3.6671, 3.6369, 3.6362, 3.6361, 3.636 , 3.6358,
             3.6358, 3.6361, 3.6361, 3.6366, 3.6374, 3.6378, 3.6378, 3.6384, 3.6396, 3.6396,
             3.6398, 3.641 , 3.6412, 3.6421, 3.6476, 3.6756, 3.667 , 3.6524, 3.6502, 3.6475,
             3.6441, 3.6417, 3.621 , 3.6204, 3.5937, 3.5931, 3.5665, 3.5659, 3.5392, 3.5386,
             3.5119, 3.5113, 3.4846, 3.484 , 3.4573, 3.4568, 3.4301, 3.4295, 3.4028, 3.4022,
             3.3814, 3.3798, 3.3789, 3.379 , 3.3799, 3.3949, 3.3552, 3.349 , 3.3427, 3.3347,
             3.3313, 3.2814, 3.2785, 3.2754, 3.2733, 3.273 , 3.2101, 3.2051, 3.1961, 3.1912,
             3.1861, 3.1364, 3.1321, 3.1297, 3.1282, 3.1278, 3.0213, 3.0031, 3.0008, 2.8686,
             2.8612, 2.8602, 2.8523, 2.7754, 2.7667, 2.7652, 2.7574, 2.6788, 2.6699, 2.6681,
             2.6586, 2.5777, 2.5715, 2.5703, 2.5604, 2.48  , 2.4746, 2.4732, 2.4658, 2.3884,
             2.3812, 2.3799, 2.3742, 2.298 , 2.2934, 2.2922, 2.2875, 2.1954, 2.1942, 2.1707,
             2.1613, 2.1824, 2.1657, 2.1653, 2.149 , 2.1486, 2.1323, 2.1318, 2.1156, 2.1152,
             2.0989, 2.0985, 2.0822, 2.0818, 2.0681, 2.0675, 2.0547, 2.0544, 2.0415, 2.0412,
             2.0283, 2.028 , 2.0151, 2.0148, 2.0019, 2.0017, 1.9888, 1.9885, 1.9756, 1.9753,
             1.9612, 1.9597, 1.958 , 1.9556, 1.9533, 1.9509, 1.9314, 1.9021, 1.9286, 1.93,
             1.9299, 1.927 , 1.9258, 1.9226, 1.9207, 1.941 , 1.9428, 1.9425, 1.9273, 1.9269,
             1.9223, 1.9232, 1.9253, 1.925 , 1.9124, 1.9121, 1.9078, 1.9078, 1.91  , 1.9097,
             1.8985, 1.8981, 1.8935, 1.8935, 1.8957, 1.8957, 1.8858, 1.8855, 1.8812, 1.881,
             1.8833, 1.883 , 1.8731, 1.8728, 1.8686, 1.8683, 1.8706, 1.8704, 1.8605, 1.8603,
             1.856 , 1.8561, 1.8586, 1.8584, 1.8499, 1.8497, 1.8459, 1.8456, 1.8483, 1.8482,
             1.8419, 1.8417, 1.838 , 1.8381, 1.8411, 1.841 , 1.8373, 1.8372, 1.8337, 1.8339,
             1.8371, 1.8371, 1.836 , 1.8359, 1.8326, 1.833 , 1.8364, 1.8365, 1.8379, 1.8379,
             1.8349, 1.8353, 1.839 , 1.8391, 1.8432, 1.8433, 1.8404, 1.841 , 1.8449, 1.8451,
             1.8517, 1.8517, 1.8492, 1.8499, 1.854 , 1.8543, 1.8634, 1.8637, 1.8613, 1.8621,
             1.8664, 1.8667, 1.8785, 1.8788, 1.8767, 1.8775, 1.882 , 1.8824, 1.8968, 1.897,
             1.8952, 1.8959, 1.9009, 1.9013, 1.9183, 1.9186, 1.917 , 1.918 , 1.923 , 1.9234,
             1.9428, 1.9433, 1.942 , 1.943 , 1.9482, 1.9487, 1.9706, 1.9712, 1.9701, 1.9581,
             1.9597, 1.9598, 1.9618, 1.9617, 1.9598, 1.957 , 1.9151, 1.9787, 1.9992, 2.0012,
             2.0044, 2.007 , 2.0091, 2.0204, 2.0207, 2.0345, 2.0348, 2.0486, 2.049 , 2.0628,
             2.0631, 2.0768, 2.0772, 2.091 , 2.0913, 2.1051, 2.1055, 2.1193, 2.1196, 2.1334,
             2.1337, 2.1475, 2.1478, 2.1616, 2.1619, 2.1757, 2.176 , 2.1898, 2.1901, 2.2014,
             2.2021, 2.2016, 2.2006, 2.1995, 2.1985, 2.1778, 2.1446]

    wall_z =[-1.2749, -1.3148, -1.3314, -1.3344, -1.3344, -1.3732, -1.3807, -1.3858, -1.3928,
             -1.4003, -1.4219, -1.4314, -1.4685, -1.4718, -1.4751, -1.4768, -1.5044, -1.5164,
             -1.5921, -1.5968, -1.6006, -1.6044, -1.6082, -1.6101, -1.6428, -1.6457, -1.6495,
             -1.6533, -1.656 , -1.6896, -1.7386, -1.7386, -1.7385, -1.738 , -1.7373, -1.7362,
             -1.7349, -1.7134, -1.7121, -1.7111, -1.7103, -1.7099, -1.7097, -1.7097, -1.704,
             -1.7037, -1.7025, -1.7009, -1.6994, -1.6527, -1.6516, -1.6517, -1.6399, -1.6387,
             -1.637 , -1.5993, -1.6149, -1.6177, -1.6333, -1.636 , -1.6526, -1.6544, -1.6698,
             -1.7056, -1.7115, -1.7115, -1.7116, -1.712 , -1.7127, -1.7137, -1.7149, -1.7159,
             -1.7413, -1.743 , -1.7441, -1.745 , -1.7455, -1.7457, -1.7458, -1.7458, -1.6822,
             -1.6227, -1.6192, -1.6137, -1.61  , -1.6028, -1.5971, -1.5915, -1.5103, -1.4984,
             -1.4892, -1.4858, -1.4824, -1.4789, -1.4739, -1.4356, -1.4316, -1.4277, -1.4238,
             -1.4171, -1.3927, -1.3895, -1.3848, -1.3819, -1.3762, -1.3348, -1.3348, -1.3348,
             -1.3385, -1.3704, -1.3704, -1.2978, -1.2141, -1.212 , -1.2094, -1.2089, -1.2089,
             -1.2089, -1.183 , -1.167 , -1.1541, -1.1489, -1.1468, -1.1436, -1.1419, -1.1403,
             -1.1373, -1.1244, -1.0832, -1.0719, -1.0632, -1.0387, -1.0365, -1.034 , -1.0321,
             -1.0269, -1.0172, -0.96  , -0.9522, -0.9494, -0.9475, -0.9421, -0.9323, -0.872,
             -0.8638, -0.8607, -0.859 , -0.8532, -0.8429, -0.7817, -0.7715, -0.7681, -0.7669,
             -0.7612, -0.7503, -0.6869, -0.6763, -0.6726, -0.6712, -0.6656, -0.654 , -0.5878,
             -0.5774, -0.5734, -0.5715, -0.5656, -0.561 , -0.4851, -0.4748, -0.4706, -0.4686,
             -0.4627, -0.4578, -0.3716, -0.3698, -0.3653, -0.3633, -0.3574, -0.3526, -0.2694,
             -0.2625, -0.2578, -0.2558, -0.2499, -0.2448, -0.1582, -0.1533, -0.1484, -0.1464,
             -0.1405, -0.135 , -0.051 , -0.0426, -0.0376, -0.0355, -0.0297, -0.0246,  0.0682,
              0.0682,  0.0744,  0.0765,  0.0822,  0.0885,  0.1759,  0.1818,  0.1871,  0.1893,
              0.1937,  0.196 ,  0.2901,  0.2948,  0.3003,  0.3025,  0.308 ,  0.313 ,  0.4072,
              0.4095,  0.4136,  0.4157,  0.4239,  0.4263,  0.5195,  0.5213,  0.5265,  0.5286,
              0.5338,  0.5392,  0.6319,  0.6341,  0.6387,  0.6408,  0.6459,  0.6493,  0.743,
              0.7454,  0.7499,  0.752 ,  0.7569,  0.7623,  0.8528,  0.8598,  0.862 ,  0.8665,
              0.87  ,  0.9607,  0.9629,  0.9678,  0.97  ,  0.9743,  0.9765,  1.0667,  1.0678,
              1.0737,  1.0757,  1.08  ,  1.0829,  1.1699,  1.1713,  1.1773,  1.1792,  1.1832,
              1.186 ,  1.2182,  1.2277,  1.2358,  1.3333,  1.3355,  1.336 ,  1.3364,  1.3376,
              1.3393,  1.3409,  1.341 ,  1.3425,  1.344 ,  1.3444,  1.3444,  1.3452,  1.3464,
              1.3464,  1.3465,  1.3474,  1.3474,  1.348 ,  1.3514,  1.366 ,  1.4229,  1.407,
              1.4056,  1.405 ,  1.4056,  1.4071,  1.4259,  1.4265,  1.4507,  1.4513,  1.4756,
              1.4761,  1.5004,  1.5009,  1.5252,  1.5257,  1.55  ,  1.5505,  1.5748,  1.5754,
              1.5996,  1.6002,  1.6244,  1.625 ,  1.6439,  1.6461,  1.6487,  1.6515,  1.6538,
              1.6705,  1.7021,  1.7001,  1.7   ,  1.7018,  1.7034,  1.7381,  1.7418,  1.7468,
              1.7528,  1.7593,  1.8154,  1.8137,  1.8136,  1.8147,  1.8168,  1.8514,  1.8571,
              1.8613,  1.8666,  1.8727,  1.9174,  1.8817,  1.8827,  1.9388,  1.942 ,  1.9422,
              1.9445,  1.9672,  1.9698,  1.9701,  1.9712,  1.9821,  1.9833,  1.9834,  1.9833,
              1.9821,  1.9821,  1.982 ,  1.9804,  1.967 ,  1.9661,  1.9658,  1.9634,  1.9382,
              1.9358,  1.9352,  1.9323,  1.894 ,  1.8917,  1.891 ,  1.8876,  1.8221,  1.8213,
              1.854 ,  1.8517,  1.823 ,  1.7901,  1.7894,  1.7572,  1.7565,  1.7245,  1.7236,
              1.6915,  1.6908,  1.6586,  1.6579,  1.6258,  1.625 ,  1.5983,  1.5975,  1.5639,
              1.5631,  1.5294,  1.5286,  1.4949,  1.4942,  1.4606,  1.4598,  1.4261,  1.4253,
              1.3917,  1.3909,  1.3572,  1.3565,  1.32  ,  1.3173,  1.316 ,  1.315 ,  1.3148,
              1.3154,  1.3229,  1.2971,  1.2758,  1.2738,  1.2724,  1.2604,  1.2578,  1.2534,
              1.252 ,  1.2437,  1.2358,  1.234 ,  1.1577,  1.1559,  1.1496,  1.1474,  1.1396,
              1.1374,  1.0601,  1.0583,  1.0518,  1.049 ,  1.0413,  1.0394,  0.9615,  0.9587,
              0.9518,  0.9491,  0.9422,  0.9396,  0.8619,  0.8594,  0.8525,  0.8497,  0.842,
              0.8402,  0.7625,  0.76  ,  0.7531,  0.7503,  0.7427,  0.7412,  0.6636,  0.6617,
              0.6547,  0.6519,  0.6443,  0.6424,  0.5651,  0.5631,  0.5565,  0.5537,  0.5462,
              0.5448,  0.4667,  0.4647,  0.4579,  0.4551,  0.4476,  0.4458,  0.368 ,  0.3659,
              0.359 ,  0.3562,  0.3489,  0.347 ,  0.2697,  0.2671,  0.26  ,  0.2572,  0.25,
              0.2482,  0.1707,  0.1681,  0.1611,  0.1583,  0.1512,  0.1493,  0.0715,  0.0695,
              0.0622,  0.0595,  0.0525,  0.0504, -0.0278, -0.0292, -0.0363,  -0.039, -0.0459,
             -0.0481, -0.1251, -0.1271, -0.1346, -0.1373, -0.144 , -0.1458, -0.2234, -0.2248,
             -0.2323, -0.235 , -0.2416, -0.2434, -0.3204, -0.322 , -0.3296, -0.3322, -0.3386,
             -0.3404, -0.417 , -0.4185, -0.4261, -0.4287, -0.4349, -0.4368, -0.5122, -0.5142,
             -0.5219, -0.5244, -0.5305, -0.532 , -0.607 , -0.609 , -0.6168, -0.6241, -0.6259,
             -0.6267, -0.6577, -0.6602, -0.6629, -0.6641, -0.6772, -0.7882, -0.7796, -0.7792,
             -0.7796, -0.7811, -0.7841, -0.8114, -0.8121, -0.8455, -0.8462, -0.8795, -0.8803,
             -0.9136, -0.9143, -0.9477, -0.9484, -0.9817, -0.9825, -1.0158, -1.0167, -1.05,
             -1.0508, -1.0841, -1.0849, -1.1182, -1.1189, -1.1522, -1.153 , -1.1863, -1.187,
             -1.2204, -1.2211, -1.2484, -1.2513, -1.2541, -1.256 , -1.2573, -1.2581, -1.2668,
             -1.2749]

    wall_z = array(wall_z)
    wall_r = array(wall_r)

    if z_limit is not None:
        inds = where(wall_z <= z_limit)
        wall_z = wall_z[inds]
        wall_r = wall_r[inds]

        wall_z = append(wall_z, wall_z[0])
        wall_r = append(wall_r, wall_r[0])

    return wall_r, wall_z




def tcv_boundary():
    R = [ 1.13600e+00, 1.13559e+00, 9.70736e-01, 9.64982e-01, 6.71635e-01,
          6.66176e-01, 6.24690e-01, 6.24000e-01, 6.24000e-01, 6.24352e-01,
          6.71850e-01, 6.78969e-01, 9.64982e-01, 9.70736e-01, 1.13559e+00,
          1.13600e+00, 1.13600e+00 ]

    z = [ 5.42692e-01, 5.49879e-01, 7.46614e-01, 7.50000e-01, 7.50000e-01,
          7.45476e-01, 7.03990e-01, 6.96942e-01, -6.96455e-01, -7.03639e-01,
          -7.49469e-01, -7.50000e-01, -7.50000e-01, -7.46614e-01, -5.49879e-01,
          -5.42692e-01, 5.42692e-01 ]

    return R, z




def tcv_inner_baffle():
    R = [ 6.2400e-01, 6.2482e-01, 6.2634e-01, 6.2857e-01, 6.3140e-01, 6.7084e-01,
          6.7348e-01, 6.7522e-01, 6.7596e-01, 6.7562e-01, 6.7425e-01, 6.7193e-01,
          6.6883e-01, 6.6520e-01, 6.6144e-01, 6.5778e-01, 6.5423e-01, 6.5081e-01,
          6.4755e-01, 6.4444e-01, 6.4152e-01, 6.3878e-01, 6.3625e-01, 6.3393e-01,
          6.3183e-01, 6.2996e-01, 6.2834e-01, 6.2697e-01, 6.2585e-01, 6.2499e-01,
          6.2440e-01, 6.2407e-01, 6.2400e-01, 6.2400e-01 ]

    z = [ -2.6081e-01, -2.6467e-01, -2.6837e-01, -2.7170e-01, -2.7454e-01, -3.0868e-01,
          -3.1169e-01, -3.1529e-01, -3.1923e-01, -3.2321e-01, -3.2696e-01, -3.3022e-01,
          -3.3275e-01, -3.3443e-01, -3.3584e-01, -3.3747e-01, -3.3933e-01, -3.4145e-01,
          -3.4378e-01, -3.4632e-01, -3.4906e-01, -3.5199e-01, -3.5510e-01, -3.5838e-01,
          -3.6180e-01, -3.6535e-01, -3.6902e-01, -3.7279e-01, -3.7664e-01, -3.8056e-01,
          -3.8453e-01, -3.8853e-01, -3.8950e-01, -2.6081e-01 ]

    return R, z




def tcv_outer_baffle():
    R = [ 1.1360e+00, 1.1320e+00, 1.0358e+00, 1.0319e+00, 1.0279e+00, 9.5592e-01, 9.5347e-01,
          9.5316e-01, 9.5359e-01, 9.5480e-01, 9.5676e-01, 9.5940e-01, 9.6261e-01, 1.1116e+00,
          1.1150e+00, 1.1181e+00, 1.1212e+00, 1.1241e+00, 1.1268e+00, 1.1293e+00, 1.1316e+00,
          1.1349e+00, 1.1360e+00, 1.1360e+00 ]

    z = [ -3.4516e-01, -3.4582e-01, -3.7335e-01, -3.7411e-01, -3.7434e-01, -3.7434e-01, -3.7175e-01,
          -3.6777e-01, -3.6380e-01, -3.5999e-01, -3.5652e-01, -3.5352e-01, -3.5113e-01, -2.6508e-01,
          -2.6292e-01, -2.6048e-01, -2.5790e-01, -2.5515e-01, -2.5220e-01, -2.4907e-01, -2.4578e-01,
          -2.4121e-01, -2.4121e-01, -3.4516e-01 ]

    return R, z




def tcv_baffled_boundary():
    R = [ 1.13600e+00, 1.13559e+00, 9.70736e-01, 9.64982e-01, 6.71635e-01,
          6.66176e-01, 6.24690e-01, 6.24000e-01, 6.2400e-01, 6.2482e-01,
          6.2634e-01, 6.2857e-01, 6.3140e-01, 6.7084e-01, 6.7348e-01,
          6.7522e-01, 6.7596e-01, 6.7562e-01, 6.7425e-01, 6.7193e-01,
          6.6883e-01, 6.6520e-01, 6.6144e-01, 6.5778e-01, 6.5423e-01,
          6.5081e-01, 6.4755e-01, 6.4444e-01, 6.4152e-01, 6.3878e-01,
          6.3625e-01, 6.3393e-01, 6.3183e-01, 6.2996e-01, 6.2834e-01,
          6.2697e-01, 6.2585e-01, 6.2499e-01, 6.2440e-01, 6.2407e-01,
          6.2400e-01, 6.24000e-01, 6.24352e-01, 6.71850e-01, 6.78969e-01,
          9.64982e-01, 9.70736e-01, 1.13559e+00, 1.13600e+00, 1.1360e+00,
          1.1320e+00, 1.0358e+00, 1.0319e+00, 1.0279e+00, 9.5592e-01,
          9.5347e-01,9.5316e-01, 9.5359e-01, 9.5480e-01, 9.5676e-01,
          9.5940e-01, 9.6261e-01, 1.1116e+00, 1.1150e+00, 1.1181e+00,
          1.1212e+00, 1.1241e+00, 1.1268e+00, 1.1293e+00, 1.1316e+00,
          1.1349e+00, 1.1360e+00, 1.13600e+00 ]

    z = [ 5.42692e-01, 5.49879e-01, 7.46614e-01, 7.50000e-01, 7.50000e-01,
          7.45476e-01, 7.03990e-01, 6.96942e-01, -2.6081e-01, -2.6467e-01,
          -2.6837e-01, -2.7170e-01, -2.7454e-01, -3.0868e-01, -3.1169e-01,
          -3.1529e-01, -3.1923e-01, -3.2321e-01, -3.2696e-01, -3.3022e-01,
          -3.3275e-01, -3.3443e-01, -3.3584e-01, -3.3747e-01, -3.3933e-01,
          -3.4145e-01, -3.4378e-01, -3.4632e-01, -3.4906e-01, -3.5199e-01,
          -3.5510e-01, -3.5838e-01, -3.6180e-01, -3.6535e-01, -3.6902e-01,
          -3.7279e-01, -3.7664e-01, -3.8056e-01, -3.8453e-01, -3.8853e-01,
          -3.8950e-01,  -6.96455e-01, -7.03639e-01, -7.49469e-01, -7.50000e-01,
          -7.50000e-01, -7.46614e-01, -5.49879e-01, -5.42692e-01, -3.4516e-01,
          -3.4582e-01, -3.7335e-01, -3.7411e-01, -3.7434e-01, -3.7434e-01,
          -3.7175e-01, -3.6777e-01, -3.6380e-01, -3.5999e-01, -3.5652e-01,
          -3.5352e-01, -3.5113e-01, -2.6508e-01, -2.6292e-01, -2.6048e-01,
          -2.5790e-01, -2.5515e-01, -2.5220e-01, -2.4907e-01, -2.4578e-01,
          -2.4121e-01, -2.4121e-01, 5.42692e-01 ]

    return R, z




def tcv_baffled_boundary_low_res():
    R = [ 1.136,   0.9679,  0.671635, 0.666176, 0.62469,
          0.624,   0.624,   0.67084,  0.67522,  0.67562,
          0.67193, 0.65778, 0.64444,  0.63393,  0.62697,
          0.624,   0.624,   0.6724,   0.96789,  1.136,
          1.136,   1.0358,  1.0279,   0.95592,  0.95316,
          0.9548,  0.9594,  1.1116,   1.1241,   1.136,
          1.136 ]

    z = [ 0.549377,  0.75,      0.75,     0.745476, 0.70399,
          0.696942, -0.268135, -0.30868, -0.31529, -0.32321,
          -0.33022, -0.33747,  -0.34632, -0.35838, -0.37279,
          -0.3895,  -0.7033,   -0.75, -   0.75,    -0.54938,
          -0.34468, -0.37335,  -0.37434, -0.37434, -0.36777,
          -0.35999, -0.352984, -0.26508, -0.25515, -0.24121,
          0.549377 ]

    return R, z



if __name__ == "__main__":
    import matplotlib.pyplot as plt

    r, z = mastu_boundary()
    fig = plt.figure(figsize = (6,8))
    ax = fig.add_subplot(111)
    ax.plot(r, z)
    ax.axis('equal')
    ax.set_xlabel('R (m)', fontsize = 13)
    ax.set_ylabel('z (m)', fontsize = 13)
    plt.tight_layout()
    plt.show()