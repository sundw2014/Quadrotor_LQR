# 3D Control of Quadcopter
# based on https://github.com/juanmed/quadrotor_sim/blob/master/3D_Quadrotor/3D_control_with_body_drag.py

import argparse
import numpy as np
import scipy
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from ol_dynamics import g, f, A, B


def lqr(A, B, Q, R):
    """Solve the continuous time lqr controller.
    dx/dt = A x + B u
    cost = integral x.T*Q*x + u.T*R*u
    """
    # http://www.mwm.im/lqr-controllers-with-python/
    # ref Bertsekas, p.151

    # first, try to solve the ricatti equation
    X = np.matrix(scipy.linalg.solve_continuous_are(A, B, Q, R))

    # compute the LQR gain
    K = np.matrix(scipy.linalg.inv(R) * (B.T * X))

    eigVals, eigVecs = scipy.linalg.eig(A - B * K)

    return np.asarray(K), np.asarray(X), np.asarray(eigVals)


####################### solve LQR #######################
n = A.shape[0]
m = B.shape[1]
Q = np.eye(n)
Q[0, 0] = 10.
Q[1, 1] = 10.
Q[2, 2] = 10.
R = np.diag([1., 1., 1.])
K, _, _ = lqr(A, B, Q, R)
# K[np.abs(K) < 1e-5] = 0
# print('K = ')
# print(K)

##### helper functions for simulation ##########
def tracking_controller(ref, t):
    # ref: T x 3
    assert ref.shape[0] == len(t)
    def u(x, _t):
        # the controller
        dis = _t - t
        dis[dis < 0] = np.inf
        idx = dis.argmin()
        return K.dot(np.array(ref[idx, :].reshape(-1).tolist()+[0, ] * 5) - x)
    return u


def simulate_linear(X0, ref, t):
    def cl_dynamics(x, t, u):
        # closed-loop dynamics. u should be a function
        x = np.array(x)
        dot_x = A.dot(x) + B.dot(u(x, t))
        return dot_x
    u = tracking_controller(ref, t)
    x_l = odeint(cl_dynamics, X0, t, args=(u,))
    return x_l


def simulate_nonlinear(X0, ref, t):
    def cl_dynamics(x, t, u):
        # closed-loop dynamics. u should be a function
        x = np.array(x)
        dot_x = f(x, u(x, t))
        return dot_x
    u = tracking_controller(ref, t)
    x_nl = odeint(cl_dynamics, X0, t, args=(u,))
    return x_nl

def simulate(X0, ref, t):
    xref = simulate_nonlinear(X0, ref, t)
    u = tracking_controller(ref, t)
    uref = np.array([u(x, t) for (x, t) in zip(xref, t)])
    return xref, uref


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='3D Quadcopter linear controller simulation')
    parser.add_argument(
        '-T',
        type=float,
        help='Total simulation time',
        default=10.0)
    parser.add_argument(
        '--time_step',
        type=float,
        help='Time step simulation',
        default=0.01)
    parser.add_argument(
        '-w', '--waypoints', type=float, nargs='+', action='append',
        help='Waypoints')
    parser.add_argument('--seed', help='seed', type=int, default=1024)
    args = parser.parse_args()

    np.random.seed(args.seed)



    ######################## simulate #######################
    # time instants for simulation
    t_max = args.T
    t = np.arange(0., t_max, args.time_step)

    if args.waypoints:
        # follow waypoints
        signal = np.zeros([len(t), 3])
        num_w = len(args.waypoints)
        for i, w in enumerate(args.waypoints):
            assert len(w) == 3
            signal[len(t) // num_w * i:len(t) // num_w *
                   (i + 1), :] = np.array(w).reshape(1, -1)
        X0 = np.zeros(8)
    else:
        # Create an random signal to track
        num_dim = 3
        freqs = np.arange(0.1, 1., 0.1)
        weights = np.random.randn(len(freqs), num_dim)  # F x n
        weights = weights / \
            np.sqrt((weights**2).sum(axis=0, keepdims=True))  # F x n
        signal_AC = np.sin(freqs.reshape(1, -1) * t.reshape(-1, 1)
                           ).dot(weights)  # T x F * F x n = T x n
        signal_DC = np.random.randn(num_dim).reshape(1, -1)  # offset
        signal = signal_AC + signal_DC
        signal[:, 2] = 0.1 * t
        # initial state
        _X0 = 0.1 * np.random.randn(num_dim) + signal_DC.reshape(-1)
        X0 = np.zeros(8)
        X0[:3] = _X0

    x_l = simulate_linear(X0, signal, t)
    x_nl = simulate_nonlinear(X0, signal, t)

    ######################## plot #######################
    signalx = signal[:, 0]
    signaly = signal[:, 1]
    signalz = signal[:, 2]

    fig = plt.figure(figsize=(20, 10))
    track = fig.add_subplot(1, 2, 1, projection="3d")
    errors = fig.add_subplot(1, 2, 2)

    track.plot(x_l[:, 0], x_l[:, 1], x_l[:, 2], color="r", label="linear")
    track.plot(x_nl[:, 0], x_nl[:, 1], x_nl[:, 2], color="g", label="nonlinear")
    if args.waypoints:
        for w in args.waypoints:
            track.plot(w[0:1], w[1:2], w[2:3], 'ro', markersize=10.)
    else:
        track.text(signalx[0], signaly[0], signalz[0], "start", color='red')
        track.text(signalx[-1], signaly[-1], signalz[-1], "finish", color='red')
        track.plot(signalx, signaly, signalz, color="b", label="command")
    track.set_title(
        "Closed Loop response with LQR Controller to random input signal {3D}")
    track.set_xlabel('x')
    track.set_ylabel('y')
    track.set_zlabel('z')
    track.legend(loc='lower left', shadow=True, fontsize='small')

    errors.plot(t, signalx - x_l[:, 0], color="r", label='x error (linear)')
    errors.plot(t, signaly - x_l[:, 1], color="g", label='y error (linear)')
    errors.plot(t, signalz - x_l[:, 2], color="b", label='z error (linear)')

    errors.plot(t, signalx - x_nl[:, 0], linestyle='-.',
                color='firebrick', label="x error (nonlinear)")
    errors.plot(t, signaly - x_nl[:, 1], linestyle='-.',
                color='mediumseagreen', label="y error (nonlinear)")
    errors.plot(t, signalz - x_nl[:, 2], linestyle='-.',
                color='royalblue', label="z error (nonlinear)")

    errors.set_title("Position error for reference tracking")
    errors.set_xlabel("time {s}")
    errors.set_ylabel("error")
    errors.legend(loc='lower right', shadow=True, fontsize='small')

    plt.show()
