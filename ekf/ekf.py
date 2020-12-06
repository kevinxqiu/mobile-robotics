# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 22:35:07 2020

@author: kevin
"""
import os
import sys
import math

sys.path.insert(0, os.path.join(os.getcwd(), 'camera'))

import get_video
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as Rot


# Covariance for EKF simulation
Q = np.diag([
    0.0001,  # variance of location on x-axis
    0.0001,  # variance of location on y-axis
    np.deg2rad(0.0001),  # variance of yaw angle
    3.45831 ** 2,  # variance of velocity mm^2/s^2
    0.0402859 ** 2 # variance of angular velocity rad^2/s^2(yaw rate)
    ]) ** 2  # predict state covariance
R = np.diag([
    0.0001,
    0.0001,
    np.deg2rad(0.0001),
    0.0001,
    0.0001
    ]) ** 2  # Observation x,y, theta position and v,w covariance

#  Simulation parameter
#INPUT_NOISE = np.diag([0.0001, np.deg2rad(0.0001)]) ** 2
#GPS_NOISE = np.diag([0.0001, 0.0001, 0.0001]) ** 2

DT = 0.1 # time tick [s]
SIM_TIME = 15  # simulation time [s]

show_animation = True

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
    x_{t+1} = x_t+v*dt*cos(yaw)
    y_{t+1} = y_t+v*dt*sin(yaw)
    yaw_{t+1} = yaw_t+omega*dt
    v_{t+1} = v{t}
    so
    dx/dyaw = -v*dt*sin(yaw)
    dx/dv = dt*cos(yaw)
    dy/dyaw = v*dt*cos(yaw)
    dy/dv = dt*sin(yaw)
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

    # add noise to gps x-y
    #measure camera
    camera_pos = get_video.detect_thymio(img,pixel2mmx,pixel2mmy)
    z = observation_model(camera_pos)

    # add noise to input
    #ud = u #+ INPUT_NOISE @ np.random.randn(2, 1)

    #xDR = motion_model(xDR, ud)

    return z, xTrue#, xDR, ud


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


"""def main():
    print(__file__ + " start!!")

    time = 0.0

    # State Vector [x y yaw v]'
    xEst = np.zeros((4, 1))
    xTrue = np.zeros((4, 1))
    PEst = np.eye(4)

    xDR = np.zeros((4, 1))  # Dead reckoning

    # history
    hxEst = xEst
    hxTrue = xTrue
    hxDR = xTrue
    hz = np.zeros((2, 1))

    while SIM_TIME >= time:
        time += DT
        u = calc_input(leftv, rightv) #ADD MOTOR ODOMETRY

        xTrue, z, xDR, ud = observation(xTrue, xDR, u)

        xEst, PEst = ekf_estimation(xEst, PEst, z, ud)

        # store data history
        hxEst = np.hstack((hxEst, xEst))
        hxDR = np.hstack((hxDR, xDR))
        hxTrue = np.hstack((hxTrue, xTrue))
        hz = np.hstack((hz, z))

        if show_animation:
            plt.cla()
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect('key_release_event',
                    lambda event: [exit(0) if event.key == 'escape' else None])
            plt.plot(hz[0, :], hz[1, :], ".g")
            plt.plot(hxTrue[0, :].flatten(),
                     hxTrue[1, :].flatten(), "-b")
            #plt.plot(hxDR[0, :].flatten(),
            #         hxDR[1, :].flatten(), "-k")
            plt.plot(hxEst[0, :].flatten(),
                     hxEst[1, :].flatten(), "-r")
            #plot_covariance_ellipse(xEst, PEst)
            #plt.axis("equal")
            plt.grid(True)
            plt.xlim(0, 0.2)
            plt.ylim(-0.05,0.05)

            plt.xlabel('x')
            plt.ylabel('y')
            plt.pause(0.001)"""


#if __name__ == '__main__':
#    main()
