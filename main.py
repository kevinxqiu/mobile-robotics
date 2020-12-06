
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
pts = get_corners(frame) # will be used to unwarp all images from live feed
#print(pts)
warped = unwarp.four_point_transform(frame, pts)
gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

# Map size is 1188 x 840
X,Y = gray.shape # X and Y are flipped here
pixel2mmx = 840 / X
pixel2mmy = 1188 / Y

pos, ang = get_video.detect_thymio(gray,pixel2mmx,pixel2mmy)

# Change to true if you want to save an image of the map
save_img = False
if save_img:
    # show and save the warped image
    #cv2.imshow("Map1", warped)
    cv2.imwrite('map2.jpg',warped)



# Start video capture saving
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output_vid.avi', fourcc, 20.0, (Y,X))

'''
RUN PATH PLANNING
'''
# start and goal position
# Map size is 1188 x 840
start = np.array([pos[0], pos[1]]).astype(int)
end = np.array([1020, 200]).astype(int)

img = 'map2.jpg'
gray = cv2.imread(img, cv2.IMREAD_GRAYSCALE)

show_animation = True
path  = voronoi_road_map.get_path(gray,show_animation,start,end)


'''
INITIALIZE THYMIO
'''
#/dev/cu.usbmodem141101	/dev/cu.usbmodem141401
th = Thymio.serial(port='COM5', refreshing_rate=0.1) #/dev/ttyACM0 for linux
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

def video_cam():
    ret, frame = cap.read()
    warped = unwarp.four_point_transform(frame, pts)
    #corners, ids, rejectedImgPoints = aruco.detectMarkers(warped, aruco_dict)
    #corners = np.array(corners)
    
    #Y,X = img.shape
    # Plot identified markers, if any
    #if ids is not None:
    #    frame_markers = aruco.drawDetectedMarkers(img, corners, ids,borderColor = (0,0,255))
    
    cv2.imshow('Image',warped)
    out.write(warped)
    cv2.waitKey()
    
    
def repeated_function():
    #global curr_speed, left, right
    global xEst, xTrue, PEst, hxEst, hxTrue, hz
    #xDR,
    #hxDR,
    
    ret, frame = cap.read()
    warped = unwarp.four_point_transform(frame, pts)
    # Print original live video feed

    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY) #converts color image to gray space
    
    #measure speed from thymio
    curr_speed = my_th.get_speed()
    left, right = curr_speed[0], curr_speed[1]
    #calculate velocity input
    u = ekf.calc_input(left, right)
    #measure position from camera
    xTrue, z = ekf.observation(xTrue, u, gray)
    #run EKF to estimate position
    xEst, PEst = ekf.ekf_estimation(xEst, PEst, z, u)
    #print(xEst)
    #correct position from estimate
    my_th.set_position([xEst[0][0], xEst[1][0], np.rad2deg(xEst[2][0])])
    #print(' position:', my_th.get_position())
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
    
    # color= cv2.cvtColor(warped,cv2.COLOR_GRAY2RGB)
    
    #line_thickness = 2
    #cv2.line(warped, (x1, y1), (x2, y2), (0, 255, 0), thickness=line_thickness)

    
rt_motion = RepeatedTimer(0.1, repeated_function)
rt_cam = RepeatedTimer(0.1, video_cam)

#add map readings
my_th.set_speed(100)

# count = 0
# for point in path:
#     if my_th.flag_skip == True and count <= 1:
#         count += 1
#         continue
#         print('hi')
#     else:
#         my_th.flag_skip = False
#         count = 0
#         my_th.move_to_target([point[0],point[1]]) 
#         print(' position:', my_th.get_position())


for point in path:
    if my_th.flag_skip > 0:
        print('skipping point!')
        my_th.flag_skip -= 1
                        
        continue

    my_th.move_to_target([point[0],point[1]]) 
    print(' position:', my_th.get_position())

#time.sleep(100000)
#my_th.move_to_target([137, 200])
#my_th.move_to_target([550, 200])

# my_th.go_straight(100)
# my_th.turn(90)
# my_th.go_straight(100)

print(path)
rt_motion.stop()
my_th.stop()
cap.release()
out.release()

cv2.destroyAllWindows()
