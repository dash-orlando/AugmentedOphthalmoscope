'''
# Program to calibrate camera and adjust paramaters for reading distance
# This routine is part of the Ophthalmoscope project.
#
# Obtains a reference circle and uses it as a scale for every other circle
# to obtain distance
#
# Author: Mohammad Odeh
# Date: Frb. 21st, 2017
# Reference: PyImageSearch
'''
# import the necessary packages
import numpy as np
import cv2
import imutils
 
def find_marker(image):
        # Convert into grayscale because HoughCircle only accepts grayscale images
        bgr2gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        bgr2gray = cv2.bilateralFilter(bgr2gray,11,17,17)

        # Threshold any color that is not black to white
        retval, thresholded = cv2.threshold(bgr2gray, 30, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
        bgr2gray = cv2.erode(cv2.dilate(thresholded, kernel, iterations=1), kernel, iterations=1)

        # find the contours in the edged image and keep the largest one;
        # we'll assume that this is our piece of paper in the image
        cnts = cv2.findContours(bgr2gray.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]

        # loop over the contours
        for c in cnts:
                # compute the center of the contour
                M = cv2.moments(c)
                if M['m00'] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])

        return (cX, cY)
 
def distance_to_camera(knownWidth, focalLength, perWidth):
	# compute and return the distance from the maker to the camera
	return (knownWidth * focalLength) / perWidth
 
# initialize the known distance from the camera to the object, which
# in this case is 24 inches
KNOWN_DISTANCE = 3.
 
# initialize the known object width, which in this case, the piece of
# paper is 11 inches wide
KNOWN_WIDTH = 1.8
 
# initialize the list of images that we'll be using
IMAGE_PATHS = ["images/3in.png"]
 
# load the furst image that contains an object that is KNOWN TO BE 2 feet
# from our camera, then find the paper marker in the image, and initialize
# the focal length
image = cv2.imread(IMAGE_PATHS[0])
marker = find_marker(image)
focalLength = (marker[0] * KNOWN_DISTANCE) / KNOWN_WIDTH

# loop over the images
for imagePath in IMAGE_PATHS:
    
    # load the image, find the marker in the image, then compute the
    # distance to the marker from the camera
    image = cv2.imread(imagePath, cv2.IMREAD_UNCHANGED)
    marker = find_marker(image)

    if imagePath == "images/3ft.jpg":
        KNOWN_WIDTH= 8.5
        
    inches = distance_to_camera(KNOWN_WIDTH, focalLength, marker[0])

    # draw a bounding box around the image and display it
    
    if imagePath == "images/3ft.jpg":
        image = cv2.resize(image, (w,h))

    image = cv2.resize(image, (936,714))
    cv2.putText(image, "%.2fft" % (inches / 12),
            (image.shape[1] - 200, image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,
            2.0, (0, 255, 0), 3)
    cv2.imshow("image", image)
    cv2.waitKey(0)
