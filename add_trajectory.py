# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 10:09:05 2020

@author: Celinna

Purpose: To overlap path trajectory onto live recorded videos

Input: Reads map.jpg and the existing videos
Output: Videos with trajectories plotted

"""
import cv2
import os
import sys
import numpy as np

# Adding the src folder in the current directory as it contains the script
# with the Thymio class
sys.path.insert(0, os.path.join(os.getcwd(), 'utils'))
sys.path.insert(0, os.path.join(os.getcwd(), 'ekf'))
sys.path.insert(0, os.path.join(os.getcwd(), 'camera'))
sys.path.insert(0, os.path.join(os.getcwd(), 'global_nav'))

import voronoi_road_map

# Get path
# Read saved map image
img = 'map2.jpg'
gray = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
#gray = cv2.flip(gray, 0)

start = np.array([130, 680]).astype(int)
end = np.array([1020, 200]).astype(int)

show_animation = True # Shows voronoi path planning process
path  = voronoi_road_map.get_path(gray,show_animation,start,end)

# Map size is 1188 x 840
X,Y = gray.shape # X and Y are flipped here
pixel2mmx = 840 / X
pixel2mmy = 1188 / Y

for point in path:
    point[0] = int(point[0] / pixel2mmy)
    point[1] = int(X-(point[1] / pixel2mmx))

r,c = path.shape

# Read image
cap = cv2.VideoCapture('kidnapping.avi')

# Save new image 
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('kidnapping_with_path.avi', fourcc, 20.0, (int(cap.get(3)),int(cap.get(4))))


while(True):
    ret, frame = cap.read()
    
    line_thickness = 2
    
    if ret == True:
        new = frame.copy()
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        for i in range(0,r-1):    
            cv2.line(new, (path[i,0], path[i,1]), (path[i+1,0], path[i+1,1]), (0, 0, 255), thickness=line_thickness)
        
        kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
        new = cv2.filter2D(new, -1, kernel)
        
        cv2.imshow('frame',new)
        out.write(new)
    
    else:
        break
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break   


cap.release()
out.release()

cv2.destroyAllWindows()