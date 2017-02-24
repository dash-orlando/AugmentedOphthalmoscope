'''
#
# Program to calibrate camera and adjust paramaters for reading distance
# This routine is part of the Ophthalmoscope project.
#
# Obtains a reference circle and uses it as a scale for every other circle
# to obtain distance for overlay activation.
#
# Author: Mohammad Odeh
# Date: Frb. 21st, 2017
# Reference: PyImageSearch
#
#
# CHANGELOG:
#    1- Routine is now better at detecting cirlces and is more reliable at distance calculations
#    2- Created a reference image scale in conjunction with the camera to be used as to get
#       better distance readings (camera is out of focus due to the nature of its application
#       thus the reference image needs to be taken at the that focus level).
#
'''

# import the necessary packages
import numpy
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

    # Find (in future update, the largest) circle outline
    circles = cv2.HoughCircles(bgr2gray, cv2.HOUGH_GRADIENT, 9, 800,
                               191, 43, 30, 35)
    # For debugging purposes only
    cv2.imshow("thresholded",cv2.resize(bgr2gray.copy(), (936,714)))

    if circles is not None:
            circles = numpy.round(circles[0,:]).astype("int")
            for (x, y, r) in circles:
                    circle=(x,y,r)
                    return(circle)

        
def distance_to_camera(knownWidth, focalLength, perWidth):
    # Compute and return the distance from the object to the camera
    return (knownWidth * focalLength) / perWidth

# Initialize the known distance from object to camera
# Calibration stand and buck were designed to have h = 3.5inches
KNOWN_DISTANCE = 3.5
 
# Initialize the known width of the scale buck
# Diamater, d = 2.0inches
KNOWN_WIDTH = 2.0
 
# Load the images we are going to use
IMAGE_PATHS = ["images/3.5inch.png"]
 
# Load the reference scale image and obtain the required
# parameters (focal length) from it by using the known variables
image = cv2.imread(IMAGE_PATHS[0])
marker = find_marker(image)
focalLength = (marker[2] * KNOWN_DISTANCE) / KNOWN_WIDTH

# Used for development purposes
# Loop over the images
for imagePath in IMAGE_PATHS:

    # Load the image, find the marker (circles) in the image, then
    # compute the distance to the marker from the camera
    image = cv2.imread(imagePath, cv2.IMREAD_UNCHANGED)
    marker = find_marker(image)
        
    inches = distance_to_camera(KNOWN_WIDTH, focalLength, marker[2])

    # Draw detected circle and print distance
    cv2.circle(image, (marker[0], marker[1]), marker[2],(0,255,0),4)
    image = cv2.resize(image, (936,714))
    cv2.putText(image, "%.2fft" % (inches / 12),
            (image.shape[1] - 200, image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,
            2.0, (0, 255, 0), 3)
    # Display Image
    cv2.imshow("image", image)
    cv2.waitKey(0)
