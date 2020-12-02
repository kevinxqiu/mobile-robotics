
import cv2
import cv2.aruco as aruco
from time import sleep
import os
import sys
import time
import serial
import math
import numpy as np

# Adding the src folder in the current directory as it contains the script
# with the Thymio class
sys.path.insert(0, os.path.join(os.getcwd(), 'utils'))
sys.path.insert(0, os.path.join(os.getcwd(), 'ekf'))
sys.path.insert(0, os.path.join(os.getcwd(), 'camera'))

#print(os.getcwd())
import unwarp
from get_corners import get_corners
import get_video
from Thymio import Thymio
from motion import Robot
from motion import RepeatedTimer
from ekf import *
from IPython.display import clear_output

cap = cv2.VideoCapture(1) # might not be 1, depending on computer
pts = get_video.init_video(cap, save_img=False)

#/dev/cu.usbmodem141101	/dev/cu.usbmodem141401
th = Thymio.serial(port="/dev/cu.usbmodem141401", refreshing_rate=0.1) #/dev/ttyACM0 for linux
my_th = Robot(th)

time.sleep(3) # To make sure the Thymio has had time to connect

variables = th.variable_description()
print(variables[0])

# %% codecell

# State Vector [x y yaw v]'

xEst = np.zeros((5, 1))
xTrue = np.zeros((5, 1))
PEst = np.eye(5)
#xDR = np.zeros((5, 1))  # Dead reckoning

# history
hxEst = xEst
hxTrue = xTrue
#hxDR = xTrue
hz = np.zeros((3, 1))

def repeated_function():
    #global curr_speed, left, right
    global xEst, xTrue, PEst, hxEst, hxTrue, hz
    #xDR,
    #hxDR,

    curr_speed = my_th.get_speed()
    left, right = curr_speed[0], curr_speed[1]

    u = calc_input(left, right)

    xTrue, z = observation(xTrue, u)
    xEst, PEst = ekf_estimation(xEst, PEst, z, u)

    my_th.set_position([xEst[0][0], xEst[1][0], np.rad2deg(xEst[2][0])])
    print('My position:', my_th.get_position())

    # store data history
    hxEst = np.hstack((hxEst, xEst))
    #hxDR = np.hstack((hxDR, xDR))
    hxTrue = np.hstack((hxTrue, xTrue))
    hz = np.hstack((hz, z))

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
    plt.axis('equal')
    plt.grid(True)
    #plt.xlim(0, 0.2)
    #plt.ylim(-0.05,0.05)

    plt.xlabel('x')
    plt.ylabel('y')
    #plt.pause(0.001)

rt_motion = RepeatedTimer(0.1, repeated_function)

#add map readings
my_th.set_speed(100)
my_th.set_position([0,0,0])
my_th.move_to_target([0,50])
my_th.move_to_target([50,50])
my_th.move_to_target([50,100])
my_th.move_to_target([70,100])
my_th.move_to_target([80,140])

print('My position:', my_th.get_position())

# my_th.go_straight(200)
# my_th.turn(45)
# my_th.go_straight(100)
# my_th.turn(90)
# my_th.go_straight(100)

rt_motion.stop()
my_th.stop()
cap.release()
cv2.destroyAllWindows()
