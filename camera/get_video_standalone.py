# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 14:20:52 2020

@author: Celinna

DEPRECIATED CODE - NO LONGER USED
"""

import numpy as np
import cv2
import cv2.aruco as aruco
from time import sleep
import unwarp
from get_corners import get_corners
import math

aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)

h = 480
w = 640

def detect_thymio(frame,pixel2mmx,pixel2mmy):
    """
        Detect the location and angle of the Thymio robot using Aruco markers
        Parameters:
            image (grayscale img): source image
            pixel2mmx (float): pixel to millimeter for x direction
            pixel2mmy (float): pixel to millimeter for y direction
        Returns:
            pos (array): [x,y] coordinates
            ang (float): angle in radians
    """
    # Detect aruco markers
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #converts color image to gray space
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict)
    corners = np.array(corners)
    
    Y, X = gray.shape
   
    # Plot identified markers, if any
    if ids is not None:
        frame_markers = aruco.drawDetectedMarkers(frame, corners, ids,borderColor = (0,0,255)) # convert back to RGBA
       
        pos = np.empty((len(ids),2))
        vec = np.empty((len(ids),2))
        
        c = corners[0][0] # Find center of marker
            
        pos = np.array([c[:,0].mean()*pixel2mmy,  (Y-c[:,1].mean())*pixel2mmx])
        pos = pos.astype(int)
        
        vec = (c[1,:] + c[2,:])/2 - (c[0,:] + c[3,:])/2 # Find orientation vector of chair
        vec = vec/ np.linalg.norm(vec) # Convert to unit vector
        ang = math.atan2(vec[1], vec[0])
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