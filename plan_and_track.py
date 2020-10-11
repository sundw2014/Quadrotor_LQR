# This file is subject to the terms and conditions defined in
# file 'LICENSE', which is part of this source code package.
import numpy as np

from src.rrt.rrt_star import RRTStar
from src.search_space.search_space import SearchSpace

import LQR
import C3M


def interpolate(wp_in):
    dt = 0.01
    v = 1.
    wp_out = []
    t = []
    for i in range(len(wp_in)-1):
        p1 = wp_in[i]
        p2 = wp_in[i+1]
        dist = np.sqrt(((p2 - p1)**2).sum())
        unit = (p2 - p1) / dist
        T = dist / v
        local_t = np.arange(0., T, dt)
        wp_out += [p1+lt*v*unit for lt in local_t]
        t += (local_t+t[-1]+dt).tolist()
    t.append(t[-1]+dt)
    wp_out.append(wp_in[-1])
    return wp_out, t


def plan(x_init, x_goal, x_bound, obstacles)
    Q = np.array([(8, 4)])  # length of tree edges
    r = 1  # length of smallest edge to check for intersection with obstacles
    max_samples = 1024  # max number of samples to take before timing out
    rewire_count = 32  # optional, number of nearby branches to rewire
    prc = 0.1  # probability of checking for a connection to goal

    # create Search Space
    X = SearchSpace(x_bound, obstacles)

    # create rrt_search
    rrt = RRTStar(X, Q, x_init, x_goal, max_samples, r, prc, rewire_count)
    path = rrt.rrt_star()

    return np.array(path) # N x 3

class Controller(object):
    """docstring for Controller"""
    def __init__(self, C3M_file, x_init=None, x_goal=None, x_bound=None, obstacles=None):
        super(Controller, self).__init__()
        self.C3M = C3M.get_model(C3M_file)
        self.x_init = None
        self.x_goal = None
        self.x_bound = None
        self.obstacles = None

        self.reset(x_init, x_goal, x_bound, obstacles)

    def reset(self, x_init, x_goal, x_bound, obstacles):
        self.x_init = x_init if x_init
        self.x_goal = x_goal if x_goal
        self.x_bound = x_bound if x_bound
        self.obstacles = obstacles if obstacles
        if self.x_init and self.x_goal and self.x_bound and self.obstacles:
            path = plan(self.x_init, self.x_goal, self.x_bound, self.obstacles)
            self.waypoints, self.t = interpolate(path)
            self.xref, self.uref = LQR.simulate(x_init, self.waypoints, self.t)

    def __call__(self, xcurr):
        dist = ((xcurr.reshape(1,-1)[:, :3] - self.xref[:, :3])**2).sum(axis=1)
        idx = dist.argmin()
        xref = self.xref[idx, :]
        uref = self.uref[idx, :]
        xe = xcurr - xref
        u = self.C3M(xcurr, xe, uref)
        return u

if __name__ == '__main__':
    import scipy
    from scipy.integrate import odeint
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from ol_dynamics import g, f, A, B

    x_init = (-9., -9, -9)  # starting location
    x_goal = ( 9.,  9,  9)  # goal location
    x_bound = np.array([(-10., 10), (-10, 10), (-10, 10)])  # dimensions of Search Space
    obstacles = np.array(
        [(2, 2, 2, 4, 4, 4), (2, 2, 6, 4, 4, 8), (2, 6, 2, 4, 8, 4), (6, 6, 2, 8, 8, 4),
         (6, 2, 2, 8, 4, 4), (6, 2, 6, 8, 4, 8), (2, 6, 6, 4, 8, 8), (6, 6, 6, 8, 8, 8)])

    controller = Controller('data/model.pth', x_init, x_goal, x_bound, obstacles)

    def simulate(X0, t):
        def cl_dynamics(x, t, u):
            # closed-loop dynamics. u should be a function
            x = np.array(x)
            dot_x = f(x, u(x))
            return dot_x
        x_nl = odeint(cl_dynamics, X0, t, args=(controller,))
        return x_nl

    x = simulate(x_init + np.random.randn(3), controller.t)
    xref = controller.xref
    waypoints = np.array(controller.waypoints)

    ######################## plot #######################

    fig = plt.figure(figsize=(20, 10))
    track = fig.add_subplot(1, 1, 1, projection="3d")

    track.plot(x[:, 0], x[:, 1], x[:, 2], color="r", label="x")
    track.plot(xref[:, 0], xref[:, 1], xref[:, 2], color="b", label="xref")
    track.plot(waypoints[:, 0], waypoints[:, 1], waypoints[:, 2], color="g", label="waypoints")

    track.set_xlabel('x')
    track.set_ylabel('y')
    track.set_zlabel('z')
    track.legend(loc='lower left', shadow=True, fontsize='small')

    plt.show()
