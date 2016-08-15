'''
* VERSION: 0.4
*   - Multithreading the raspberry picam now allows for a higher framerate
*
* KNOWN ISSUES:
*   - Circle detection needs a lot of improvement and filtering
*   - I/O call must be moved into an entire separate thread to further
*     increase FPS and decrease the effects of I/O latency
'''

print __doc__

from imutils.video.pivideostream import PiVideoStream
from imutils.video import FPS
from picamera.array import PiRGBArray
from picamera import PiCamera
import SimpleCV
import time
import cv2
import numpy

# Setup camera
vs = PiVideoStream().start()
normalDisplay = True
time.sleep(2)

while True:
    # Get image from stream
    image = vs.read()
    
    # Convert into grayscale because HoughCircle only accepts grayscale images
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Reduce Noise
    grayscale = cv2.medianBlur(grayscale,5)
    grayscale = cv2.GaussianBlur(grayscale,(5,5),0)

    # Adaptive threshold allows us to detect sharp edges in images
    grayscale = cv2.adaptiveThreshold(grayscale,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,3.5)

    # Scan for circles
    circles = cv2.HoughCircles(grayscale, cv2.cv.CV_HOUGH_GRADIENT,1.2,100,
                               param1=55,param2=50,minRadius=50,maxRadius=100)

    # If circles are found draw them
    if circles is not None:
        circles = numpy.round(circles[0,:]).astype("int")
        for (x, y, r) in circles:
            # Draw circle
            cv2.circle(image, (x,y),r,(0,255,0),4)

    # Nothing important for the time being
    if normalDisplay:
        cv2.namedWindow("Live Feed ver0.4")
        cv2.imshow("Live Feed ver0.4", image)
        key = cv2.waitKey(1) & 0xFF
