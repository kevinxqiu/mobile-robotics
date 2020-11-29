# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 16:38:11 2020

@author: Celinna
"""

import cv2
import numpy as np

def get_corners(img):
    # read image
    #img = cv2.imread("sample-map.jpg")
    #img = cv2.medianBlur(img,5)
    #img = cv2.GaussianBlur(img,(5,5),0)
    #img = cv2.bilateralFilter(img,9,75,75)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    _, blackAndWhite = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    
    # convert img to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = 255-gray
    
    # do adaptive threshold on gray image
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 17, 1)
    thresh = 255-thresh
    
    # apply morphology
    kernel = np.ones((3,3), np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel)
    
    # separate horizontal and vertical lines to filter out spots outside the rectangle
    kernel = np.ones((7,3), np.uint8)
    vert = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)
    kernel = np.ones((3,7), np.uint8)
    horiz = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)
    
    # combine
    rect = cv2.add(horiz,vert)
    
    # thin
    kernel = np.ones((3,3), np.uint8)
    rect = cv2.morphologyEx(rect, cv2.MORPH_ERODE, kernel)
    
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(rect, None, None, None, 8, cv2.CV_32S)
    sizes = stats[1:, -1] #get CC_STAT_AREA component
    img2 = np.zeros((labels.shape), np.uint8)
    
    for i in range(0, nlabels - 1):
        if sizes[i] >= 210:   #filter small dotted regions
            img2[labels == i + 1] = 255
    
    res = img2
    #res = cv2.bitwise_not(img2)
    #invert = cv2.bitwise_not(image)
    
    
    # get largest contour
    contours = cv2.findContours(res, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    for c in contours:
        area_thresh = 0
        area = cv2.contourArea(c)
        if area > area_thresh:
            area = area_thresh
            big_contour = c
    
    
    # define main island contour approx. and hull
    #perimeter = cv2.arcLength(big_contour,True)
    epsilon = 0.01*cv2.arcLength(big_contour,True)
    approx = cv2.approxPolyDP(big_contour,epsilon,True)
    approx = np.reshape(approx,(4,2))
    print(approx)
    
    # get rotated rectangle from contour
    rot_rect = cv2.minAreaRect(big_contour)
    box = cv2.boxPoints(rot_rect)
    box = np.int0(box)
    #print(box)
    
    # draw rotated rectangle on copy of img
    rot_bbox = img.copy()
    cv2.drawContours(rot_bbox,[box],0,(0,0,255),2)
    save = False
    
    if save:
        # write img with red rotated bounding box to disk
        cv2.imwrite("rectangle_thresh.png", thresh)
        cv2.imwrite("rectangle_outline.png", rect)
        cv2.imwrite("rectangle_bounds.png", rot_bbox)
    
    show = False
    if show:
    # display it
        cv2.imshow('remove spots',res)
        cv2.imshow("IMAGE", img)
        #cv2.imshow("THRESHOLD", thresh)
        #cv2.imshow("MORPH", morph)
        #cv2.imshow("VERT", vert)
        #cv2.imshow("HORIZ", horiz)
        cv2.imshow("RECT", rect)
        cv2.imshow("BBOX", rot_bbox)
        cv2.waitKey(0)
    
    return approx