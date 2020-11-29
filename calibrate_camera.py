# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 12:25:14 2020

Code for displaying image from Kinect retrieved from https://github.com/r9y9/pylibfreenect2/blob/master/examples/selective_streams.py

freenect2 commands here https://rjw57.github.io/freenect2-python/


@author: Celinna
"""

# coding: utf-8

# An example using startStreams

import numpy as np
import cv2
import cv2.aruco as aruco
import sys
from time import sleep



'''INITIATE ARUCO MARKER DICTIONARY'''
aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
generate_aruco_board = False

board = aruco.CharucoBoard_create(7, 5, 1, .8, aruco_dict)

# Print the board for calibration
if generate_aruco_board == True:
    
    imboard = board.draw((2000, 2000))
    cv2.imwrite('charuco.png',imboard)


def calibrate_camera(allCorners,allIds,imsize):
    """
    Calibrates the camera using the dected corners.
    """
    print("CAMERA CALIBRATION")

    cameraMatrixInit = np.array([[ 1000.,    0., imsize[0]/2.],
                                 [    0., 1000., imsize[1]/2.],
                                 [    0.,    0.,           1.]])

    distCoeffsInit = np.zeros((5,1))
    flags = (cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_RATIONAL_MODEL + cv2.CALIB_FIX_ASPECT_RATIO)
    #flags = (cv2.CALIB_RATIONAL_MODEL)
    (ret, camera_matrix, distortion_coefficients0,rotation_vectors, translation_vectors,stdDeviationsIntrinsics, stdDeviationsExtrinsics, perViewErrors) = cv2.aruco.calibrateCameraCharucoExtended(charucoCorners=allCorners,
    charucoIds=allIds,
    board=board,
    imageSize=imsize,
    cameraMatrix=cameraMatrixInit,
    distCoeffs=distCoeffsInit,
    flags=flags,
    criteria=(cv2.TERM_CRITERIA_EPS & cv2.TERM_CRITERIA_COUNT, 10000, 1e-9))

    return ret, camera_matrix, distortion_coefficients0, rotation_vectors, translation_vectors


"""
MAIN LOOP
- Camera listener will continuously update image feed
- Aruco ID detection done for each frame
- Orientation of chair determined
- Detection of person
- Motor commands determined & sent to roombots
"""

allCorners = []
allIds = []
decimator = 0

# SUB PIXEL CORNER DETECTION CRITERION
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.00001)


#=============================================================================================================================================================================================='
#   MAIN LOOP
#=============================================================================================================================================================================================='
cap = cv2.VideoCapture(1) # might not be 1, depending on computer

for i in range(100):
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame = cv2.flip(frame,0)
    frame = cv2.flip(frame,1)
    
    # Detect aruco markers
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #converts color image to gray space
    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, aruco_dict)
    corners = np.array(corners)
    
    print("=> Processing image", str(i))
    if len(corners)>0:
        # SUB PIXEL DETECTION
        for corner in corners:
            cv2.cornerSubPix(gray, corner,
                                winSize = (3,3),
                                zeroZone = (-1,-1),
                                criteria = criteria)
        res2 = cv2.aruco.interpolateCornersCharuco(corners,ids,gray,board)
        if res2[1] is not None and res2[2] is not None and len(res2[1])>3 and decimator%1==0:
            allCorners.append(res2[1])
            allIds.append(res2[2])

    if ids is not None:
        frame_markers = aruco.drawDetectedMarkers(frame, corners, ids,borderColor = (0,0,255))
    
    decimator+=1
    sleep(0.01)
    #print(ids)
    #print(rejectedImgPoints)

    # Display image
    cv2.imshow("color", frame)
        

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



imsize = gray.shape
ret, mtx, dist, rvecs, tvecs = calibrate_camera(allCorners,allIds,imsize)

print(mtx)
print(dist)


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
