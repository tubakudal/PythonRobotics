"""
Inverted Pendulum LQR control
author: Trung Kien - letrungkien.k53.hut@gmail.com
"""

import math
import time

import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv, eig

# Model parameters

l_bar = 2.0  # length of bar
M = 1.0  # mass of the cart [kg]
m = 0.3  # mass of the pendulum [kg]
g = 9.8  # acceleration due to gravity [m/s^2]

nx = 4  # number of state variables
nu = 1  # number of input variables
Q = np.diag([0.0, 1.0, 1.0, 0.0])  # state cost matrix
R = np.diag([0.01])  # input cost matrix

delta_t = 0.1  # time step [s]
sim_time = 5.0  # simulation duration [s]

show_animation = True  # flag to show animation


def main():
    x0 = np.array([
        [0.0],  # initial position of the cart
        [0.0],  # initial velocity of the cart
        [0.3],  # initial angle of the pendulum
        [0.0]   # initial angular velocity of the pendulum
    ])

    x = np.copy(x0)
    time = 0.0

    while sim_time > time:
        time += delta_t

        # calculate control input
        u = lqr_control(x)

        # simulate inverted pendulum cart
        x = simulation(x, u)

        if show_animation:
            plt.clf()
            px = float(x[0, 0])  # cart position
            theta = float(x[2, 0])  # pendulum angle
            plot_cart(px, theta)
            plt.xlim([-5.0, 2.0])
            plt.pause(0.001)

    print("Finish")
    print(f"x={float(x[0, 0]):.2f} [m] , theta={math.degrees(x[2, 0]):.2f} [deg]")
    if show_animation:
        plt.show()


def simulation(x, u):
    A, B = get_model_matrix()
    x = A @ x + B @ u  # update state

    return x


def solve_DARE(A, B, Q, R, maxiter=150, eps=0.01):
    """
    Solve a discrete-time Algebraic Riccati equation (DARE)
    """
    P = Q

    for i in range(maxiter):
        Pn = A.T @ P @ A - A.T @ P @ B @ \
            inv(R + B.T @ P @ B) @ B.T @ P @ A + Q
        if (abs(Pn - P)).max() < eps:  # convergence check
            break
        P = Pn

    return Pn


def dlqr(A, B, Q, R):
    """
    Solve the discrete-time LQR controller.
    x[k+1] = A x[k] + B u[k]
    cost = sum x[k].T*Q*x[k] + u[k].T*R*u[k]
    # ref Bertsekas, p.151
    """

    # first, try to solve the Riccati equation
    P = solve_DARE(A, B, Q, R)

    # compute the LQR gain
    K = inv(B.T @ P @ B + R) @ (B.T @ P @ A)

    eigVals, eigVecs = eig(A - B @ K)  # compute eigenvalues for stability check
    return K, P, eigVals


def lqr_control(x):
    A, B = get_model_matrix()
    start = time.time()
    K, _, _ = dlqr(A, B, Q, R)
    u = -K @ x  # control input calculation
    elapsed_time = time.time() - start
    print(f"calc time:{elapsed_time:.6f} [sec]")
    return u


def get_numpy_array_from_matrix(x):
    """
    get built-in list from matrix
    """
    return np.array(x).flatten()


def get_model_matrix():
    A = np.array([
        [0.0, 1.0, 0.0, 0.0],  # state transition matrix
        [0.0, 0.0, m * g / M, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, g * (M + m) / (l_bar * M), 0.0]
    ])
    A = np.eye(nx) + delta_t * A  # discretize A

    B = np.array([
        [0.0],
        [1.0 / M],
        [0.0],
        [1.0 / (l_bar * M)]
    ])
    B = delta_t * B  # discretize B

    return A, B


def flatten(a):
    return np.array(a).flatten()


def plot_cart(xt, theta):
    cart_w = 1.0  # cart width
    cart_h = 0.5  # cart height
    radius = 0.1  # wheel radius

    cx = np.array([-cart_w / 2.0, cart_w / 2.0, cart_w /
                   2.0, -cart_w / 2.0, -cart_w / 2.0])
    cy = np.array([0.0, 0.0, cart_h, cart_h, 0.0])
    cy += radius * 2.0

    cx = cx + xt  # cart position

    bx = np.array([0.0, l_bar * math.sin(-theta)])  # pendulum position
    bx += xt
    by = np.array([cart_h, l_bar * math.cos(-theta) + cart_h])
    by += radius * 2.0

    angles = np.arange(0.0, math.pi * 2.0, math.radians(3.0))
    ox = np.array([radius * math.cos(a) for a in angles])
    oy = np.array([radius * math.sin(a) for a in angles])

    rwx = np.copy(ox) + cart_w / 4.0 + xt  # right wheel position
    rwy = np.copy(oy) + radius
    lwx = np.copy(ox) - cart_w / 4.0 + xt  # left wheel position
    lwy = np.copy(oy) + radius

    wx = np.copy(ox) + bx[-1]  # pendulum end position
    wy = np.copy(oy) + by[-1]

    plt.plot(flatten(cx), flatten(cy), "-b")
    plt.plot(flatten(bx), flatten(by), "-k")
    plt.plot(flatten(rwx), flatten(rwy), "-k")
    plt.plot(flatten(lwx), flatten(lwy), "-k")
    plt.plot(flatten(wx), flatten(wy), "-k")
    plt.title(f"x: {xt:.2f} , theta: {math.degrees(theta):.2f}")

    # for stopping simulation with the esc key.
    plt.gcf().canvas.mpl_connect(
        'key_release_event',
        lambda event: [exit(0) if event.key == 'escape' else None])

    plt.axis("equal")


if __name__ == '__main__':
    main()
