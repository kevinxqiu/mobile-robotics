# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 16:28:17 2020

@author: Celinna
"""

import cv2
from matplotlib import pyplot as plt

# Read image via OpenCV
img = cv2.imread('warped-img.jpg')

print(img.shape)
# Attention: OpenCV uses BGR color ordering per default whereas
# Matplotlib assumes RGB color ordering!
plt.figure(1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()