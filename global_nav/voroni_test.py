# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 17:32:02 2020

@author: Celinna
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

im = 'warped-img.jpg'

def get_map():
    pixel2mmx = 2.5
    pixel2mmy = 2.1
    factor = 25
    
    gray = cv2.imread('warped-img.jpg', cv2.IMREAD_GRAYSCALE)
    row, col = gray.shape[:2]
    #bottom = gray[row-2:row, 0:col]
    #mean = cv2.mean(bottom)[0]
    
    bordersize = 10
    border = cv2.copyMakeBorder(
        gray,
        top=bordersize,
        bottom=bordersize,
        left=bordersize,
        right=bordersize,
        borderType=cv2.BORDER_CONSTANT,
        value=[0, 0, 0]
    )
    
    #plt.imshow(border)
    # cv2.imshow('image', img)
    # cv2.imshow('bottom', bottom)
    # cv2.imshow('border', border)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    h,w = gray.shape
    #plt.imshow(gray)
    
    new_img = cv2.resize(border,(int(pixel2mmx*w/factor), int(pixel2mmy*h/factor))) 
    
    ret, thresh = cv2.threshold(new_img,127,255,cv2.THRESH_BINARY_INV)

    #thresh = np.asarray(thresh)
    thresh = np.rot90(thresh,k=1, axes=(1,0))
    print(thresh.shape)
    
    coords = np.column_stack(np.where(thresh == 255))
    coords = np.transpose(coords)
    
    ox = coords[0,:]
    oy = coords[1,:]
    
    #print(coords)
    #plt.imshow(thresh)
    #plt.imshow(coords)
    #print(coords.shape)
    #print(ox.shape)
    
    return ox, oy


ox, oy = get_map()

show_animation = True

# start and goal position
sx = 10.0  # [m]
sy = 10.0  # [m]
gx = 50.0  # [m]
gy = 50.0  # [m]
robot_size = 2.0  # [m]
    

if show_animation:  # pragma: no cover
    plt.plot(ox, oy, ".k")
    plt.plot(sx, sy, "^r")
    plt.plot(gx, gy, "^c")
    plt.grid(True)
    plt.axis("equal")

