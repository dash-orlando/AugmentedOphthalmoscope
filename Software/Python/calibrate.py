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
#    1- Reduced memory footprint
#    2- Tweaked HoughCircles parameters
#    3- Print thresholded array to file for debugging purposes
#
'''

# import the necessary packages
import numpy
import cv2
import imutils

import  sys
import  os
from    os.path                import expanduser
 
def find_marker(image):

    # Convert into grayscale because HoughCircle only accepts grayscale images
    bgr2gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bgr2gray = cv2.bilateralFilter(bgr2gray,11,17,17)

    # Threshold any color that is not black to white
    retval, thresholded = cv2.threshold(bgr2gray, 20, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    bgr2gray = cv2.erode(cv2.dilate(thresholded, kernel, iterations=1), kernel, iterations=1)
    print(bgr2gray.shape)

    # Debugging Purposes
    # ___START___
    '''    
    retval, binaryInv = cv2.threshold(bgr2gray, 50, 255, cv2.THRESH_BINARY_INV)
    kernelBinary = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    binaryInv = cv2.erode(cv2.dilate(binaryInv, kernelBinary, iterations=1), kernelBinary, iterations=1)

    retval, trunc = cv2.threshold(bgr2gray, 50, 255, cv2.THRESH_TRUNC)
    kernelTrunc = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    trunc = cv2.erode(cv2.dilate(trunc, kernelTrunc, iterations=1), kernelTrunc, iterations=1)
    
    retval, zero = cv2.threshold(bgr2gray, 50, 255, cv2.THRESH_TOZERO)
    kernelZero = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    zero = cv2.erode(cv2.dilate(zero, kernelZero, iterations=1), kernelZero, iterations=1)
    
    retval, zeroInv = cv2.threshold(bgr2gray, 50, 255, cv2.THRESH_TOZERO_INV)
    kernelZeroInv = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    zeroInv = cv2.erode(cv2.dilate(zeroInv, kernelZeroInv, iterations=1), kernelZeroInv, iterations=1)

    titles = ['binary','binaryInv','truncate','tozero','tozeroInv']
    images = [thresholded, binaryInv, trunc, zero, zeroInv]

    for i in range(1,5):
        cv2.imshow(titles[i], cv2.resize(images[i], (936, 714)))
    '''
    
    # Print data on device-specific text files
    print("Writing array to datafile...\nPlease wait...")
    rows = bgr2gray.shape[0]
    columns = bgr2gray.shape[1]
    array=bgr2gray.copy()

    dataFileDir = "/home/pi/Desktop" 

    if os.path.exists(dataFileDir) == False:
        os.makedirs(dataFileDir)

    dataFileName = "/numpyArray.txt"
    dataFilePath = dataFileDir + dataFileName

    for i in range(rows):
        for j in range(columns):
            if array[i][j]==255:
                array[i][j]=1
       

    for i in range(rows):
        for j in range(columns):
            with open(dataFilePath, "a") as dataFile:
                dataFile.write(str(array[i][j]))
        with open(dataFilePath, "a") as dataFile:
            dataFile.write('\n')
    print("Done!!")
    # ___END___
    
    
    # Find circle outline
    circles = cv2.HoughCircles(bgr2gray, cv2.HOUGH_GRADIENT, 14, 396,
                               191, 43, 50, 70)
    '''
    ORIGINAL:
    circles = cv2.HoughCircles(bgr2gray, cv2.HOUGH_GRADIENT, 16, 1000,
                               191, 43, 50, 150)
    '''
    # For debugging purposes only
    cv2.imshow("thresholded", bgr2gray.copy())

    if circles is not None:
            circles = numpy.round(circles[0,:]).astype("int")
            print("Shape: ")
            print(circles.shape)
            print("Array content: ")
            print(circles)
            for (x, y, r) in circles:
                circle=(x,y,r)
                pos=str(circle)
                cv2.circle(image, (x, y), r,(0,255,0),4)
                cv2.putText(image, pos, (x - 20, y - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            cv2.imshow("image", image)
            cv2.waitKey(0)
            return(x,y,r)

        
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
image = cv2.resize(cv2.imread(IMAGE_PATHS[0]), (360, 276))
marker = find_marker(image)
focalLength = (marker[2] * KNOWN_DISTANCE) / KNOWN_WIDTH

# Used for development purposes
# Loop over the images
for imagePath in IMAGE_PATHS:

    # Load the image, find the marker (circles) in the image, then
    # compute the distance to the marker from the camera
    image = cv2.resize(cv2.imread(imagePath, cv2.IMREAD_UNCHANGED), (360, 276))
    marker = find_marker(image)
        
    inches = distance_to_camera(KNOWN_WIDTH, focalLength, marker[2])

    # Draw detected circle and print distance
    cv2.circle(image, (marker[0], marker[1]), marker[2],(0,255,0),4)
    cv2.putText(image, "%.2fft" % (inches / 12),
            (image.shape[1] - 100, image.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
            1.0, (0, 255, 0), 3)
    # Display Image
    cv2.imshow("image", image)
    cv2.waitKey(0)
