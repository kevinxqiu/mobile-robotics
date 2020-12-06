# -*- coding: utf-8 -*-

import cv2
from matplotlib import pyplot as plt

# Read image via OpenCV
img = cv2.imread('get_goal.jpg')
crop_img = img[275:330,450:520]
# flip = img[::-1,:,:]
#print(img.shape)
# Attention: OpenCV uses BGR color ordering per default whereas
# Matplotlib assumes RGB color ordering!
# cv2.imshow("cropped", crop_img)
# cv2.waitKey(0)
cv2.imread('template.jpg')
#cv2.imwrite('template.jpg',crop_img)
plt.figure(1)

plt.imshow(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
# plt.gca().invert_yaxis()
plt.show()