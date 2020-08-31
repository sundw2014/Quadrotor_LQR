# 3D Control of Quadcopter
# based on https://github.com/juanmed/quadrotor_sim/blob/master/3D_Quadrotor/3D_control_with_body_drag.py
# The dynamics is from pp. 17, Eq. (2.22). https://www.kth.se/polopoly_fs/1.588039.1550155544!/Thesis%20KTH%20-%20Francesco%20Sabatino.pdf
# The linearization is from Different Linearization Control Techniques for
# a Quadrotor System (many typos)

import nonlinear_dynamics
import argparse
import numpy as np
import scipy
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from nonlinear_dynamics import g, m, Ix, Iy, Iz


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


# The control can be done in a decentralized style
# The linearized system is divided into four decoupled subsystems

# X-subsystem
# The state variables are x, dot_x, pitch, dot_pitch
Ax = np.array(
    [[0.0, 1.0, 0.0, 0.0],
     [0.0, 0.0, g, 0.0],
     [0.0, 0.0, 0.0, 1.0],
     [0.0, 0.0, 0.0, 0.0]])
Bx = np.array(
    [[0.0],
     [0.0],
     [0.0],
     [1 / Ix]])

# Y-subsystem
# The state variables are y, dot_y, roll, dot_roll
Ay = np.array(
    [[0.0, 1.0, 0.0, 0.0],
     [0.0, 0.0, -g, 0.0],
     [0.0, 0.0, 0.0, 1.0],
     [0.0, 0.0, 0.0, 0.0]])
By = np.array(
    [[0.0],
     [0.0],
     [0.0],
     [1 / Iy]])

# Z-subsystem
# The state variables are z, dot_z
Az = np.array(
    [[0.0, 1.0],
     [0.0, 0.0]])
Bz = np.array(
    [[0.0],
     [1 / m]])

# Yaw-subsystem
# The state variables are yaw, dot_yaw
Ayaw = np.array(
    [[0.0, 1.0],
     [0.0, 0.0]])
Byaw = np.array(
    [[0.0],
     [1 / Iz]])

####################### solve LQR #######################
Ks = []  # feedback gain matrices K for each subsystem
for A, B in ((Ax, Bx), (Ay, By), (Az, Bz), (Ayaw, Byaw)):
    n = A.shape[0]
    m = B.shape[1]
    Q = np.eye(n)
    Q[0, 0] = 10.  # The first state variable is the one we care about.
    R = np.diag([1., ])
    K, _, _ = lqr(A, B, Q, R)
    Ks.append(K)

######################## simulate #######################
# time instants for simulation
t_max = args.T
t = np.arange(0., t_max, args.time_step)


def cl_linear(x, t, u):
    # closed-loop dynamics. u should be a function
    x = np.array(x)
    X, Y, Z, Yaw = x[[0, 1, 8, 9]], x[[2, 3, 6, 7]], x[[4, 5]], x[[10, 11]]
    UZ, UY, UX, UYaw = u(x, t).reshape(-1).tolist()
    dot_X = Ax.dot(X) + (Bx * UX).reshape(-1)
    dot_Y = Ay.dot(Y) + (By * UY).reshape(-1)
    dot_Z = Az.dot(Z) + (Bz * UZ).reshape(-1)
    dot_Yaw = Ayaw.dot(Yaw) + (Byaw * UYaw).reshape(-1)
    dot_x = np.concatenate(
        [dot_X[[0, 1]], dot_Y[[0, 1]], dot_Z, dot_Y[[2, 3]], dot_X[[2, 3]], dot_Yaw])
    return dot_x


def cl_nonlinear(x, t, u):
    x = np.array(x)
    dot_x = nonlinear_dynamics.f(x, u(x, t) + np.array([m * g, 0, 0, 0]))
    return dot_x


if args.waypoints:
    # follow waypoints
    signal = np.zeros([len(t), 3])
    num_w = len(args.waypoints)
    for i, w in enumerate(args.waypoints):
        assert len(w) == 3
        signal[len(t) // num_w * i:len(t) // num_w *
               (i + 1), :] = np.array(w).reshape(1, -1)
    X0 = np.zeros(12)
    signalx = signal[:, 0]
    signaly = signal[:, 1]
    signalz = signal[:, 2]
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
    signalx = signal[:, 0]
    signaly = signal[:, 1]
    signalz = 0.1 * t
    # initial state
    _X0 = 0.1 * np.random.randn(num_dim) + signal_DC.reshape(-1)
    X0 = np.zeros(12)
    X0[[0, 2, 4]] = _X0

signalyaw = np.zeros_like(signalz)  # we do not care about yaw


def u(x, _t):
    # the controller
    dis = _t - t
    dis[dis < 0] = np.inf
    idx = dis.argmin()
    UX = Ks[0].dot(np.array([signalx[idx], 0, 0, 0]) - x[[0, 1, 8, 9]])[0]
    UY = Ks[1].dot(np.array([signaly[idx], 0, 0, 0]) - x[[2, 3, 6, 7]])[0]
    UZ = Ks[2].dot(np.array([signalz[idx], 0]) - x[[4, 5]])[0]
    UYaw = Ks[3].dot(np.array([signalyaw[idx], 0]) - x[[10, 11]])[0]
    return np.array([UZ, UY, UX, UYaw])


# simulate
x_l = odeint(cl_linear, X0, t, args=(u,))
x_nl = odeint(cl_nonlinear, X0, t, args=(u,))


######################## plot #######################
fig = plt.figure(figsize=(20, 10))
track = fig.add_subplot(1, 2, 1, projection="3d")
errors = fig.add_subplot(1, 2, 2)

track.plot(x_l[:, 0], x_l[:, 2], x_l[:, 4], color="r", label="linear")
track.plot(x_nl[:, 0], x_nl[:, 2], x_nl[:, 4], color="g", label="nonlinear")
track.plot(signalx, signaly, signalz, color="b", label="command")
if args.waypoints:
    for w in args.waypoints:
        track.plot(w[0:1], w[1:2], w[2:3], 'ro', markersize=10.)
else:
    track.text(signalx[0], signaly[0], signalz[0], "start", color='red')
    track.text(signalx[-1], signaly[-1], signalz[-1], "finish", color='red')
track.set_title(
    "Closed Loop response with LQR Controller to random input signal {3D}")
track.set_xlabel('x')
track.set_ylabel('y')
track.set_zlabel('z')
track.legend(loc='lower left', shadow=True, fontsize='small')

errors.plot(t, signalx - x_l[:, 0], color="r", label='x error (linear)')
errors.plot(t, signaly - x_l[:, 2], color="g", label='y error (linear)')
errors.plot(t, signalz - x_l[:, 4], color="b", label='z error (linear)')

errors.plot(t, signalx - x_nl[:, 0], linestyle='-.',
            color='firebrick', label="x error (nonlinear)")
errors.plot(t, signaly - x_nl[:, 2], linestyle='-.',
            color='mediumseagreen', label="y error (nonlinear)")
errors.plot(t, signalz - x_nl[:, 4], linestyle='-.',
            color='royalblue', label="z error (nonlinear)")

errors.set_title("Position error for reference tracking")
errors.set_xlabel("time {s}")
errors.set_ylabel("error")
errors.legend(loc='lower right', shadow=True, fontsize='small')

plt.show()
