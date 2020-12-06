# -*- coding: utf-8 -*-

import os
import sys
import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, os.path.join(os.getcwd(), 'camera'))

import get_video

def calc_input(leftv, rightv):
    L = 95 #wheelbase length in mm
    #conversion from Thymio speed to mm/s
    #leftv = leftv_thymio * 0.31573
    #rightv = rightv_thymio * 0.31573
    if bool(leftv < 0) ^ bool(rightv < 0):
        yawrate = (rightv - leftv) / L  #[rad/s]
        v = 0
    else:
        v = (rightv + leftv) / 2 #[mm/s]
        yawrate = 0

    u = np.array([[v], [yawrate]])
    #print(u)
    return u

def motion_model(x, u):
    yaw = x[2, 0]

    F = np.array([[1.0, 0, 0, 0, 0],
                  [0, 1.0, 0, 0, 0],
                  [0, 0, 1.0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0]
                  ])

    B = np.array([[DT * math.cos(yaw), 0],
                  [DT * math.sin(yaw), 0],
                  [0.0, DT],
                  [1.0, 0.0],
                  [0.0, 1.0]
                  ])

    x = F @ x + B @ u + Q @ np.random.randn(5,1)

    return x


def observation_model(x):
    H = np.array([
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1]
        ])

    z = H @ x + R @ np.random.randn(5, 1)
    return z


def jacob_f(x, u):
    """
    Jacobian of Motion Model
    motion model
    x{t+1} = x_t+v{t}*dt*cos(yaw{t})
    y{t+1} = y_t+v{t}*dt*sin(yaw{t})
    yaw{t+1} = yaw{t}+omega{t}*dt
    v{t+1} = v{t}
    omega{t+1} = omega{t}
    so
    dx/dyaw = -v{t}*dt*sin(yaw{t})
    dx/dv = dt*cos(yaw{t})
    dy/dyaw = v*dt*cos(yaw{t})
    dy/dv = dt*sin(yaw{t})
    dyaw/domega = dt
    """
    yaw = x[2, 0]
    v = u[0, 0]

    jF = np.array([
        [1.0, 0.0, -DT * v * math.sin(yaw), DT * math.cos(yaw), 0.0],
        [0.0, 1.0, DT * v * math.cos(yaw), DT * math.sin(yaw), 0.0],
        [0.0, 0.0, 1.0, 0.0, DT],
        [0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0]
        ])

    return jF

def jacob_h():
    # Jacobian of Observation Model
    jH = np.array([
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1]
        ])

    return jH


def observation(xTrue, u, img, pixel2mmx, pixel2mmy):
    xTrue = motion_model(xTrue, u)
    #measure camera
    camera_pos = get_video.detect_thymio(img,pixel2mmx,pixel2mmy)
    z = observation_model(camera_pos)
    return z, xTrue


def ekf_estimation(xEst, PEst, z, u):
    #  Predict
    xPred = motion_model(xEst, u)
    jF = jacob_f(xEst, u)
    PPred = jF @ PEst @ jF.T + Q

    #  Update
    jH = jacob_h()
    zPred = observation_model(xPred)
    y = z - zPred
    S = jH @ PPred @ jH.T + R
    K = PPred @ jH.T @ np.linalg.inv(S)
    xEst = xPred + K @ y
    PEst = (np.eye(len(xEst)) - K @ jH) @ PPred
    return xEst, PEst


def plot_covariance_ellipse(xEst, PEst):  # pragma: no cover
    Pxy = PEst[0:2, 0:2]
    eigval, eigvec = np.linalg.eig(Pxy)

    if eigval[0] >= eigval[1]:
        bigind = 0
        smallind = 1
    else:
        bigind = 1
        smallind = 0

    t = np.arange(0, 2 * math.pi + 0.1, 0.1)
    a = math.sqrt(eigval[bigind])
    b = math.sqrt(eigval[smallind])
    x = [a * math.cos(it) for it in t]
    y = [b * math.sin(it) for it in t]
    angle = math.atan2(eigvec[1, bigind], eigvec[0, bigind])
    rot = Rot.from_euler('z', angle).as_matrix()[0:2, 0:2]
    fx = rot @ (np.array([x, y]))
    px = np.array(fx[0, :] + xEst[0, 0]).flatten()
    py = np.array(fx[1, :] + xEst[1, 0]).flatten()
    plt.plot(px, py, "--r")

# Covariance for EKF simulation
Q = np.diag([
    10 ** 2,  # process variance of location on x-axis - mm^2
    10 ** 2,  # process variance of location on y-axis - mm^2
    0.17454 ** 2, # (10 degrees) process variance of yaw angle rad^2
    2.44539 ** 2,  # process variance of velocity mm^2/s^2
    0.00114760 ** 2 # process variance of angular velocity rad^2/s^2(yaw rate)
    ])  # predict state covariance
R = np.diag([
    2 ** 2,  # measurement variance of location on x-axis - mm^2
    2 ** 2,  # measurement variance of location on y-axis - mm^2
    0.0436 ** 2, # (2.5 degrees) measurement variance of yaw angle rad^2
    2.44539 ** 2,  # measurement variance of velocity mm^2/s^2
    0.00114760 ** 2 # measurement variance of angular velocity rad^2/s^2(yaw rate)
    ])  # Observation x,y, theta position and v,w covariance

#  Simulation parameter
#INPUT_NOISE = np.diag([0.0001, np.deg2rad(0.0001)]) ** 2
#GPS_NOISE = np.diag([0.0001, 0.0001, 0.0001]) ** 2

DT = 0.1 # time tick [s]
SIM_TIME = 15  # simulation time [s]

show_animation = True
