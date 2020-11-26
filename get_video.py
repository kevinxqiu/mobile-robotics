# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 14:20:52 2020

@author: Celinna
"""

import numpy as np
import cv2
import cv2.aruco as aruco


aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)

paramsAruco = aruco.DetectorParameters_create()
paramsAruco.adaptiveThreshWinSizeMin = 4
paramsAruco.adaptiveThreshWinSizeMax = 19
paramsAruco.adaptiveThreshWinSizeStep = 9
paramsAruco.minMarkerPerimeterRate = 0.005
paramsAruco.maxMarkerPerimeterRate = 2.5
paramsAruco.perspectiveRemovePixelPerCell = 9


def detect_thymio(frame):
    # Detect aruco markers
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #converts color image to gray space
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=paramsAruco)
    corners = np.array(corners)
    
    # Plot identified markers, if any
    if ids is not None:
        color_rgb = cv2.cvtColor(frame,cv2.COLOR_RGBA2RGB) # convert RGBA to RGB image for the drawDetectMarkers function
        frame_markers = aruco.drawDetectedMarkers(color_rgb, corners, ids,borderColor = (0,0,255)) # convert back to RGBA
       
        #colorArr = cv2.cvtColor(color_rgb.copy(),cv2.COLOR_RGB2RGBA) # Converts images back to RGBA 

        pos = np.empty((len(ids),2))
        vec = np.empty((len(ids),2))
        

        for i in range(len(ids)):
            c = corners[i][0] # Find center of marker
            
            pos[i,:] = [c[:,0].mean(),  c[:,1].mean()]
            pos = pos.astype(int)
            #print('Chair located at'+str(chair_x)+ " and " + str(chair_y))
        
            vec[i,:] = (c[1,:] + c[2,:])/2 - (c[0,:] + c[3,:])/2 # Find orientation vector of chair
            vec[i,:] = vec[i,:] / np.linalg.norm(vec[i,:]) # Convert to unit vector

            arrow_endpos = (int(pos[i,0]+vec[i,0]*100),int(pos[i,1]+vec[i,1]*100))
            
            cv2.arrowedLine(frame,tuple(pos[i,:]),arrow_endpos,(255,0,0),(2),8,0,0.1) # Draw chair vector on img
    else:
        pos = []
        vec = []
    
    return pos, vec



#=============================================================================================================================================================================================='
#   MAIN LOOP
#=============================================================================================================================================================================================='
cap = cv2.VideoCapture(1) # might not be 1, depending on computer

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Detect Thymio location
    pos, vec = detect_thymio(frame)

    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()