# -*- coding: utf-8 -*-
"""
Purpose: Test if video camera is working
Can also be used to save a static image from the video
"""
import numpy as np
import cv2
import unwarp
import get_corners
cap = cv2.VideoCapture(1)

while(cap.isOpened()):
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    save_img = False
    if save_img:
        cv2.imwrite('raw_map.jpg',frame)
        break
        
    #get_corners(gray)
    #pts = unwarp.warp_img(gray)
    #print(pts)
    
cap.release()
cv2.destroyAllWindows()