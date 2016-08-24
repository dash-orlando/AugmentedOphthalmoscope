'''
* VERSION: 0.6
*   - Program can successfully detect a pupil at a close range
*   - Resolution reduced to (480, 368)
*   - Added min/maxRadius trackbars to manipulate circle
*     constrains mid-session
*
* KNOWN ISSUES:
*   - Minor modification to circle detection algorithm is needed
'''

print __doc__

from imutils.video.pivideostream import PiVideoStream
from time import sleep
import cv2
import numpy

# Define placeholder function for trackbar
def placeholder(x):
    pass

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
stream = PiVideoStream(hf=True).start()
normalDisplay = True
sleep(2)

# Setup window and mouseCallback event
cv2.namedWindow("Live Feed ver0.6")
cv2.setMouseCallback("Live Feed ver0.6", control)

# Create a track bar for HoughCircles parameters
cv2.createTrackbar("minRadius", "Live Feed ver0.6", 5, 200, placeholder)
cv2.createTrackbar("maxRadius", "Live Feed ver0.6", 70, 250, placeholder)

# Infinite loop
while True:
    
    # Get image from stream
    image = stream.read()
    
    # Convert into grayscale because HoughCircle only accepts grayscale images
    bgr2gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Get trackbar position and reflect it in HoughCircles parameters input
    minRadius = cv2.getTrackbarPos("minRadius", "Live Feed ver0.6")
    maxRadius = cv2.getTrackbarPos("maxRadius", "Live Feed ver0.6")
    
    # Reduce Noise
    blur = cv2.medianBlur(bgr2gray,5)
    gaussBlur = cv2.GaussianBlur(blur,(5,5),0)

    # Adaptive threshold allows us to detect sharp edges in images
    threshold = cv2.adaptiveThreshold(gaussBlur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY,11,3.5)

    # Scan for circles
    circles = cv2.HoughCircles(threshold, cv2.HOUGH_GRADIENT, 12, 396,
                               191, 199, minRadius, maxRadius)
    '''
    Experimental values:            Original Values (kinda crappy but still):
    dp = 12                         dp = 1.2
    minDist = 396                   minDist = 100
    param1 = 191                    param1 = 55
    param2 = 199                    param2 = 50
    minRadius = 5                   minRadius = 50
    maxRadius = 70                  maxRadius = 100
    '''

    # If circles are found draw them
    if circles is not None:
        circles = numpy.round(circles[0,:]).astype("int")
        for (x, y, r) in circles:
            # Draw circle
            cv2.circle(image, (x,y),r,(0,255,0),4)

    # Live feed display toggle
    if normalDisplay:
        cv2.imshow("Live Feed ver0.6", image)
        key = cv2.waitKey(1) & 0xFF
    elif not(normalDisplay):
        cv2.imshow("Live Feed ver0.6", threshold)
        key = cv2.waitKey(1) & 0xFF
