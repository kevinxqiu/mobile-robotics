
import cv2
import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

# Adding the src folder in the current directory as it contains the script
# with the Thymio class
sys.path.insert(0, os.path.join(os.getcwd(), 'utils'))
sys.path.insert(0, os.path.join(os.getcwd(), 'ekf'))
sys.path.insert(0, os.path.join(os.getcwd(), 'camera'))
sys.path.insert(0, os.path.join(os.getcwd(), 'global_nav'))

#print(os.getcwd())
import unwarp
from get_corners import get_corners
import get_video
from Thymio import Thymio
from motion import Robot
from motion import RepeatedTimer
import ekf
#from IPython.display import clear_output
import voronoi_road_map



'''
START VIDEO
'''
cap = cv2.VideoCapture(1) # might not be 1, depending on computer

ret, frame = cap.read()

#frame = cv2.flip(frame,0) # Flip image axis for calculations, need to flip back for display

#cv2.imshow('hi',frame)
pts = get_corners(frame) # will be used to unwarp all images from live feed
#print(pts)
warped = unwarp.four_point_transform(frame, pts)

pixel2mmx = 2.56
pixel2mmy = 2.14
gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
pos, ang = get_video.detect_thymio(gray,pixel2mmx,pixel2mmy)


# save_img = True
# if save_img:
#     # show and save the warped image
#     #cv2.imshow("Map1", warped)
#     cv2.imwrite('map1.jpg',warped)

'''
RUN PATH PLANNING
'''
# # start and goal position
# # Map size is 1188 x 840
start = np.array([pos[0], pos[1]]).astype(int)
end = np.array([1050, 200]).astype(int)

img = 'map.jpg'
gray = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
# Isolate green layer
# b, g, r = cv2.split(img)
# (thresh, g) = cv2.threshold(g, 127, 255, cv2.THRESH_BINARY)

# M = cv2.moments(g)
# if M["m00"] != 0 :
#     cX = int(M["m10"] / M["m00"])
#     cY = int(M["m01"] / M["m00"])
#     start = [cX,cY]
# else:
#     start = []

#print(np.rad2deg(ang))
#print(pos)
path  = voronoi_road_map.get_path(gray,True,start,end)
# #print(path)


'''
INITIALIZE THYMIO
'''
#/dev/cu.usbmodem141101	/dev/cu.usbmodem141401
th = Thymio.serial(port='COM4', refreshing_rate=0.1) #/dev/ttyACM0 for linux
my_th = Robot(th)
my_th.set_position([start[0],start[1],np.rad2deg(ang)])

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
hz = np.zeros((5, 1))

def repeated_function():
    #global curr_speed, left, right
    global xEst, xTrue, PEst, hxEst, hxTrue, hz
    global hpos
    hpos = []
    #xDR,
    #hxDR,
    
    ret, frame = cap.read()
    warped = unwarp.four_point_transform(frame, pts)
    warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY) #converts color image to gray space
    
    #measure speed from thymio
    curr_speed = my_th.get_speed()
    left, right = curr_speed[0], curr_speed[1]
    #calculate velocity input
    u = ekf.calc_input(left, right)
    #measure position from camera
    xTrue, z = ekf.observation(xTrue, u, warped)
    #run EKF to estimate position
    xEst, PEst = ekf.ekf_estimation(xEst, PEst, z, u)
    print(xEst)
    #correct position from estimate
    my_th.set_position([xEst[0][0], xEst[1][0], np.rad2deg(xEst[2][0])])
    #print(' position:', my_th.get_position())
    hpos.append(my_th.get_position)
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
    
    #plt.show()

    # Print original live video feed
    cv2.imshow('Image',warped)

rt_motion = RepeatedTimer(0.1, repeated_function)

#add map readings
my_th.set_speed(100)

# path = np.array([[ 137, 790],
#  [ 160, 710],
#  [ 160, 630],
#  [ 160,  530],
#  [ 160,  430],
#  [ 163,  330],
#  [ 195,  230],
#  [ 250,  196],
#  [ 330,  190],
#  [ 410,  190],
#  [ 510,  191],
#  [ 580,  230],
#  [ 605,  290],
#  [ 620,  390],
#  [ 620,  450],
#  [ 634,  550],
#  [ 684,  650],
#  [ 698,  670],
#  [ 770,  700],
#  [ 790,  700],
#  [ 870,  700],
#  [ 970,  697],
#  [1030,  730],
#  [1050,  200]])

for point in path:
    my_th.move_to_target([point[0],point[1]])
    print('My position:', my_th.get_position())

#time.sleep(100000)
#my_th.move_to_target([137, 200])
#my_th.move_to_target([550, 200])

# my_th.go_straight(100)
# my_th.turn(90)
# my_th.go_straight(100)
print(hpos)

rt_motion.stop()
my_th.stop()
cap.release()
cv2.destroyAllWindows()
