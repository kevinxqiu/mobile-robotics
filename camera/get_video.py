# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 14:20:52 2020

@author: Celinna
"""

import numpy as np
import cv2
import cv2.aruco as aruco
import unwarp
from get_corners import get_corners
import math

aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)


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
    
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #converts color image to gray space
    corners, ids, rejectedImgPoints = aruco.detectMarkers(frame, aruco_dict)
    corners = np.array(corners)
    
    Y,X = frame.shape
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
        #print(c)
        vec = (c[1,:] + c[0,:])/2 - (c[2,:] + c[3,:])/2  # Find orientation vector of chair
        vec = vec/ np.linalg.norm(vec) # Convert to unit vector
        ang = math.atan2(vec[0], vec[1])
       
        #print(ang)
        #arrow_endpos = (int(pos[0]+vec[0]*100),int(pos[1]+vec[1]*100))
        #cv2.arrowedLine(frame,tuple(pos),arrow_endpos,(255,0,0),(2),8,0,0.1) # Draw chair vector on img
    else:
        pos = []
        ang = []
    
    return pos, ang


# Pixel to mmm ratio -- must double check


#=============================================================================================================================================================================================='
#   MAIN LOOP
#=============================================================================================================================================================================================='

cap = cv2.VideoCapture(1) # might not be 1, depending on computer

def init_video(cap, save_img):
    # First we get the warped image
    ret, frame = cap.read()
    pts = get_corners(frame) # will be used to unwarp all images from live feed
    #print(pts)
    
    if save_img:
        warped = unwarp.four_point_transform(frame, pts)
        # show and save the warped image
        cv2.imshow("Warped", warped)
        cv2.imwrite('warped-img.jpg',warped)
    
    return cap, pts



def get_video(warped):
    # initialize position
    newPos = np.zeros((5,1))

    Y,X = warped.shape
    #plt.imshow(gray)
    pixel2mmx = int(1188/X)
    pixel2mmy = int(840/Y)
    
    # Detect Thymio location
    pos, ang = detect_thymio(warped,pixel2mmx,pixel2mmy)
    
    if pos != []:
        newPos[0] = pos[0]
        newPos[1] = pos[1]
        newPos[2] = ang
    
    #print(newPos)
    #cv2.imwrite('sample-map.jpg',frame)
    #break
    
    #warped = cv2.resize(warped,(2*w, 2*h))
    
    #kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
    #img = cv2.filter2D(warped, -1, kernel)

    return newPos


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()