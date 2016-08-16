'''
* VERSION: 0.5
*   - Multithreading the raspberry picam now allows for a higher framerate
*
* KNOWN ISSUES:
*   - Circle detection needs improvement and filtering
*   - I/O call must be moved into a separate thread to further
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

# Define right/left mouse click events
def control(event, x, y, flags, param):
    global normalDisplay
    # Right button shuts down program
    if event == cv2.EVENT_RBUTTONDOWN:
        stream.stop()
        cv2.destroyAllWindows()
        quit()
    # Left button toggles display
    elif event == cv2.EVENT_LBUTTONDOWN:
        normalDisplay=not(normalDisplay)

# Setup camera
stream = PiVideoStream().start()
normalDisplay = True
time.sleep(2)

# Setup window and mouseCallback event
cv2.namedWindow("Live Feed ver0.5")
cv2.setMouseCallback("Live Feed ver0.5", control)

# Infinite loop
while True:
    
    # Get image from stream
    image = stream.read()
    
    # Convert into grayscale because HoughCircle only accepts grayscale images
    bgr2gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Reduce Noise
    blur = cv2.medianBlur(bgr2gray,5)
    gaussBlur = cv2.GaussianBlur(blur,(5,5),0)

    # Adaptive threshold allows us to detect sharp edges in images
    threshold = cv2.adaptiveThreshold(gaussBlur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY,11,3.5)

    # Scan for circles
    circles = cv2.HoughCircles(threshold, cv2.cv.CV_HOUGH_GRADIENT,1.2,100,
                               param1=55,param2=50,minRadius=50,maxRadius=100)

    # If circles are found draw them
    if circles is not None:
        circles = numpy.round(circles[0,:]).astype("int")
        for (x, y, r) in circles:
            # Draw circle
            cv2.circle(image, (x,y),r,(0,255,0),4)

    # Live feed display toggle
    if normalDisplay:
        cv2.imshow("Live Feed ver0.5", image)
        key = cv2.waitKey(1) & 0xFF
    elif not(normalDisplay):
        cv2.imshow("Live Feed ver0.5", threshold)
        key = cv2.waitKey(1) & 0xFF
