# This file is subject to the terms and conditions defined in
# file 'LICENSE', which is part of this source code package.
import numpy as np

from src.rrt.rrt_star import RRTStar
from src.search_space.search_space import SearchSpace

import LQR

# Obstacles = np.array(
#     [(20, 20, 20, 40, 40, 40), (20, 20, 60, 40, 40, 80), (20, 60, 20, 40, 80, 40), (60, 60, 20, 80, 80, 40),
#      (60, 20, 20, 80, 40, 40), (60, 20, 60, 80, 40, 80), (20, 60, 60, 40, 80, 80), (60, 60, 60, 80, 80, 80)])
# x_init = (0, 0, 0)  # starting location
# x_goal = (100, 100, 100)  # goal location
# x_bound = np.array([(-10, 10), (-10, 10), (-10, 10)])  # dimensions of Search Space

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
        self.C3M = model(C3M_file)
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
        xe = xref - xcurr
        u = self.C3M(xcurr, xe, uref)
        return u
