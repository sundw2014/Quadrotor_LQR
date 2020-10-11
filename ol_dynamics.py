import numpy as np

# quadrotor physical constants
g = 9.81

# non-linear dynamics
def f(x, u):
    x, y, z, vx, vy, vz, theta_x, theta_y = x.reshape(-1).tolist()
    az, omega_x, omega_y = u.reshape(-1).tolist()
    dot_x = np.array([
     vx,
     vy,
     vz,
     g * np.tan(theta_x),
     g * np.tan(theta_y),
     az,
     omega_x,
     omega_y])
    return dot_x

# linearization
# The state variables are x, y, z, vx, vy, vz, theta_x, theta_y
A = np.zeros([8,8])
A[0, 3] = 1.
A[1, 4] = 1.
A[2, 5] = 1.
A[3, 6] = g
A[4, 7] = g
B = np.zeros([8, 3])
B[5, 0] = 1.
B[6, 1] = 1.
B[7, 2] = 1.
