# -*- coding: utf-8 -*-
"""
NOTE:
The functions order_point and four_point_transform were retrieved from: 
https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/

author: Adrian Rosebrock

"""
import numpy as np
import cv2

def order_points(pts):
    """
    Order the 4 points of an object in correct order
    Parameters:
        pts(np.array): [x,y] coordinates of 4 corner points
    Returns:
        rect(np.array): ordered [x,y] coordinates of 4 corners
    """
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype = "float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    #pts = np.array(pts)
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect

def four_point_transform(image, pts):
    """
    Warp an image so that it is flat using four point transform
    Parameters:
        image(img): source image in color
        pts(np.array): [x,y] coordinates of 4 corner points
    Returns:
        warped(img): warped image in color
    """
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    # return the warped image
    return warped

def get_corners(img):
    """
    Get the corners of a rectangular object (the map) from an image of the map taken at an angle
    Parameters:
        image(img): source image in color
    Returns:
       approx(np.array): 4x2 array of [x,y] coordinates of 4 corners of the map
    """
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
    #print(approx)
    r,h,c = approx.shape
    
    if r != 4:
        print('ERROR! Could not find vertices to warp image. Make sure the entire map is shown in the image frame...')
        pass
    else:
        approx = np.reshape(approx,(4,2)) # getting verticies of map from image
        
        # get rotated rectangle from contour
        rot_rect = cv2.minAreaRect(big_contour)
        box = cv2.boxPoints(rot_rect)
        box = np.int0(box)
        #print(box)
        
        # draw rotated rectangle on copy of img
        rot_bbox = img.copy()
        cv2.drawContours(rot_bbox,[box],0,(0,0,255),2)
    
    save = True
    if save:
        # write img with red rotated bounding box to disk
        cv2.imwrite("rectangle_thresh.jpg", thresh)
        cv2.imwrite("rectangle_res.jpg", res)
        cv2.imwrite("rectangle_rect.jpg", rect)
        # cv2.imwrite("rectangle_bounds.png", rot_bbox)
    
    show = False
    if show:
    # display it
        cv2.imshow('remove spots',res)
        cv2.imshow("IMAGE", img)
        #cv2.imshow("THRESHOLD", thresh)
        cv2.imshow("RECT", rect)
        cv2.imshow("BBOX", rot_bbox)
        cv2.waitKey(0)
    
    return approx


# img = cv2.imread('raw_map.jpg')
# approx = get_corners(img)
    