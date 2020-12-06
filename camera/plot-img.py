# -*- coding: utf-8 -*-

import cv2
from matplotlib import pyplot as plt

# Read image via OpenCV
img = cv2.imread('warped-img.jpg')
flip = img[::-1,:,:]
#print(img.shape)
# Attention: OpenCV uses BGR color ordering per default whereas
# Matplotlib assumes RGB color ordering!
plt.figure(1)

plt.imshow(cv2.cvtColor(flip, cv2.COLOR_BGR2RGB))
plt.gca().invert_yaxis()
plt.show()