# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 23:22:05 2020

@author: kevin
"""
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.transform import resize

#fig, ax = plt.subplots(1,2)
def create_map(image_dir):
    threshold = 0.35
    scale = 10    
    image = plt.imread(image_dir)
    
    
    height, width, channel = image.shape
    
    
    image = resize(image, (height/scale, width/scale))
    height, width, channel = image.shape

    #print(height, width)
    gray = rgb2gray(image)
    bwimage = np.where(gray > threshold, 1, 0)
    bwimage[0,:] = 0
    bwimage[:,0] = 0
    bwimage[height-1,:] = 0
    bwimage[:,width-1] = 0
    
    ox = []
    oy = []
    for i in range(height):
        for j in range(width):
            if bwimage[i][j] == 0:
                ox.append(j)
                oy.append(i)
        
    
    #ax[0].imshow(image)
    #ax[0].set_title('image')
    #plt.imshow(bwimage, cmap = 'gray')
    #plt.title('map')
    #print(bwimage)
    #print(len(ox))
    return ox, oy, scale
#create_map('map.jpg')