# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 14:20:52 2020

@author: Celinna
"""

import numpy as np
import cv2
import cv2.aruco as aruco
from time import sleep
import unwarp
from get_corners import get_corners
import math

aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)

# Camera calibration stuff NOT USED
# mtx = np.array([[1.42136061e+03, 0.00000000e+00, 2.31190962e+02],
#  [0.00000000e+00, 1.42136061e+03, 2.69893517e+02],
#  [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

# dist = np.array([[ 3.77240855e-01],
#  [ 4.30142117e+00],
#  [ 3.51204744e-02],
#  [-7.20942014e-02],
#  [-6.00795847e+01],
#  [-2.47754984e-01],
#  [ 4.30058684e-02],
#  [-1.39645742e+01],
#  [ 0.00000000e+00],
#  [ 0.00000000e+00],
#  [ 0.00000000e+00],
#  [ 0.00000000e+00],
#  [ 0.00000000e+00],
#  [ 0.00000000e+00]])

#newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

h = 480
w = 640

def detect_thymio(frame,pixel2mmx,pixel2mmy):
    """
    Parameters
    ----------
    frame : IMAGE
        Source image
    pixel2mmx : INT
        Pixel scaling factor for x direction
    pixel2mmy : INT
        Pixel scaling factor for y direction

    Returns
    -------
    pos : INT 
        Current (x,y) position with shape (1,2)
    vec : TYPE
        Returns none

    """
    # Detect aruco markers
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #converts color image to gray space
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict)
    corners = np.array(corners)
    
    Y, X = gray.shape
   
    # Plot identified markers, if any
    if ids is not None:
        frame_markers = aruco.drawDetectedMarkers(frame, corners, ids,borderColor = (0,0,255)) # convert back to RGBA
       
        #colorArr = cv2.cvtColor(color_rgb.copy(),cv2.COLOR_RGB2RGBA) # Converts images back to RGBA 

        pos = np.empty((len(ids),2))
        vec = np.empty((len(ids),2))
        
        c = corners[0][0] # Find center of marker
            
        pos = np.array([c[:,0].mean()*pixel2mmy,  (Y-c[:,1].mean())*pixel2mmx])
        pos = pos.astype(int)
        
        #print('Chair located at'+str(chair_x)+ " and " + str(chair_y))
        
        vec = (c[1,:] + c[2,:])/2 - (c[0,:] + c[3,:])/2 # Find orientation vector of chair
        vec = vec/ np.linalg.norm(vec) # Convert to unit vector
        #print(vec)
        ang = math.atan2(vec[1], vec[0])
        #ang = np.angle(vec,deg = True)
        #print(ang)
        #arrow_endpos = (int(pos[0]+vec[0]*100),int(pos[1]+vec[1]*100))
        vec = []
        #cv2.arrowedLine(frame,tuple(pos),arrow_endpos,(255,0,0),(2),8,0,0.1) # Draw chair vector on img
    else:
        pos = []
        ang = []
    
    return pos, ang


# Pixel to mmm ratio -- must double check
pixel2mmx = 2.5
pixel2mmy = 2.1

#=============================================================================================================================================================================================='
#   MAIN LOOP
#=============================================================================================================================================================================================='
cap = cv2.VideoCapture(1) # might not be 1, depending on computer

# First we get the warped image
ret, frame = cap.read()
pts = get_corners(frame) # will be used to unwarp all images from live feed

#print(pts)

#warped = unwarp.four_point_transform(frame, pts)
# show and save the warped image
#cv2.imshow("Warped", warped)
#cv2.imwrite('warped-img.jpg',warped)

# initialize position
newPos = np.zeros((4,1))

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    #frame = cv2.flip(frame,0)
    
     # Get warped iamge
    warped = unwarp.four_point_transform(frame, pts)
    # Detect Thymio location
    pos, ang = detect_thymio(warped,pixel2mmx,pixel2mmy)

    
    if pos.any():
        newPos[0] = pos[0].astype(int)
        newPos[1] = pos[1].astype(int)
        newPos[2] = ang
    
    print(newPos)
    #cv2.imwrite('sample-map.jpg',frame)
    #break
    
    #warped = cv2.resize(warped,(2*w, 2*h))
    
    kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
    img = cv2.filter2D(warped, -1, kernel)

    # Print original live video feed
    cv2.imshow('Image',img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
   
    sleep(0.1)


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()