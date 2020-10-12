import numpy as np
import sys
sys.path.append('rrt-algorithms/')

from src.rrt.rrt_star import RRTStar
from src.search_space.search_space import SearchSpace

import LQR
import C3M

def interpolate(wp_in):
    dt = 0.001 # FIXME
    v = 1. # FIXME
    wp_out = []
    t = []
    currtent_t = 0.
    for i in range(wp_in.shape[0]-1):
        p1 = wp_in[i, :]
        p2 = wp_in[i+1, :]
        dist = np.sqrt(((p2 - p1)**2).sum())
        unit = (p2 - p1) / dist
        T = dist / v
        local_t = np.arange(0., T, dt)
        wp_out += [p1+lt*v*unit for lt in local_t]
        t += (local_t+currtent_t).tolist()
        currtent_t = t[-1] + dt
    t.append(currtent_t)
    wp_out.append(wp_in[-1,:])
    return np.array(wp_out), np.array(t)


def plan(x_init, x_goal, x_bound, obstacles):
    import random
    random.seed(0)

    Q = np.array([(8, 4)])  # length of tree edges
    r = 1  # length of smallest edge to check for intersection with obstacles
    max_samples = 1024  # max number of samples to take before timing out
    rewire_count = 32  # optional, number of nearby branches to rewire
    prc = 0.1  # probability of checking for a connection to goal

    # create Search Space
    X = SearchSpace(x_bound, obstacles)

    # create rrt_search
    rrt = RRTStar(X, Q, tuple(x_init), tuple(x_goal), max_samples, r, prc, rewire_count)
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
        if x_init is not None: self.x_init = x_init
        if x_goal is not None: self.x_goal = x_goal
        if x_bound is not None: self.x_bound = x_bound
        if obstacles is not None: self.obstacles = obstacles
        if all(v is not None for v in [self.x_init, self.x_goal, self.x_bound, self.obstacles]):
            path = plan(self.x_init[:3], self.x_goal, self.x_bound, self.obstacles)
            self.waypoints, self.t = interpolate(path)
            self.xref, self.uref = LQR.simulate(self.x_init, self.waypoints, self.t)

    def __call__(self, xcurr):
        dist = ((xcurr.reshape(1,-1)[:, :3] - self.xref[:, :3])**2).sum(axis=1)
        idx = dist.argmin()
        idx = idx + int(0.1 / (self.t[1] - self.t[0])) # look ahead for 0.1 second
        idx = idx if idx < self.xref.shape[0] else self.xref.shape[0]-1
        xref = self.xref[idx, :]
        uref = self.uref[idx, :]
        xe = xcurr - xref
        u = self.C3M(xcurr, xe, uref)
        return u

if __name__ == '__main__':
    import scipy
    from LQR import odeint
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from ol_dynamics import g, f, A, B

    x_init = [0., 0, 0] + [0.,]*5  # starting location and other state variables
    x_goal = (9.,  9,  9)  # goal location
    x_bound = np.array([(-10., 10), (-10, 10), (-10, 10)])  # dimensions of Search Space
    obstacles = np.array(
        [(2, 2, 2, 4, 4, 4), (2, 2, 6, 4, 4, 8), (2, 6, 2, 4, 8, 4), (6, 6, 2, 8, 8, 4),
         (6, 2, 2, 8, 4, 4), (6, 2, 6, 8, 4, 8), (2, 6, 6, 4, 8, 8), (6, 6, 6, 8, 8, 8)])

    controller = Controller('data/model_best.pth.tar', x_init, x_goal, x_bound, obstacles)


    def simulate(X0, t):
        def cl_dynamics(x, t, u):
            # closed-loop dynamics. u should be a function
            x = np.array(x)
            dot_x = f(x, u(x))
            return dot_x
        x_nl = odeint(cl_dynamics, X0, t, args=(controller,))
        return x_nl

    x = simulate(np.array(x_init) + np.concatenate([1.*np.random.randn(3), np.random.randn(5)]), controller.t)
    xref = controller.xref
    waypoints = np.array(controller.waypoints)

    ######################## plot #######################
    fig = plt.figure(figsize=(20, 10))
    track = fig.add_subplot(1, 1, 1, projection="3d")

    track.plot(waypoints[:, 0], waypoints[:, 1], waypoints[:, 2], color="g", label="waypoints")
    track.plot(xref[:, 0], xref[:, 1], xref[:, 2], color="b", label="xref")
    track.plot(x[:, 0], x[:, 1], x[:, 2], color="r", label="x")

    print(x[-1, :])
    track.set_xlabel('x')
    track.set_ylabel('y')
    track.set_zlabel('z')
    track.legend(loc='lower left', shadow=True, fontsize='small')

    plt.show()
