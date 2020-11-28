# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 23:22:05 2020

@author: kevin
"""
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray

def get_map(image):
    fig, ax = plt.subplots(1,2)
    threshold = 0.35
    
    image = plt.imread(image)
    gray = rgb2gray(image)
    bwimage = np.where(gray > threshold, 0, 1)
    
    ax[0].imshow(image)
    ax[0].set_title('image')
    ax[1].imshow(bwimage, cmap = 'gray')
    ax[1].set_title('map')
    
    return bwimage, ax