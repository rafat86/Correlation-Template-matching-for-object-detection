# Assignment 03 #
# Correlation: Template matching for object detection #
# Ra'fat Naserdeen #

import cv2
import numpy as np

full_image = cv2.imread('clutter.jpg')
template = cv2.imread('mug.png')

# 01 Convert your images to gray
gray_image = cv2.cvtColor(full_image, cv2.COLOR_BGR2GRAY)
gray_temp = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

# 02 Compute the correlation of the image and the template
res = cv2.matchTemplate(gray_image, gray_temp, cv2.TM_CCOEFF_NORMED)

# 03 Keep only the values that exceed some threshold value in res
threshold = 0.9
res_threshold = np.where(res < threshold, 0, res)


# 04 Write a function BBox to draw a bounding box around the detected objects
def draw_bbox(gray_temp, gray_image, threshold):
    locations = np.where(res >= threshold)
    temp_height = gray_temp.shape[1]
    temp_width = gray_temp.shape[0]
    for loc in zip(*locations[::-1]):
        top_left = loc
        bottom_right = (top_left[0] + temp_height, top_left[1] + temp_width)
        cv2.rectangle(gray_image, top_left, bottom_right, 255, 1)

    return cv2.imshow('Gray Image', gray_image)


draw_bbox(gray_temp, gray_image, threshold)
cv2.imshow('template', gray_temp)
cv2.imshow('res', res)
cv2.waitKey(0)
cv2.destroyAllWindows()
