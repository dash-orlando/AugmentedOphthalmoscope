'''
* VERSION: 0.6
*   - Program can successfully detect a pupil at a close range
*   - Resolution reduced to (480, 368)
*   - Added trackbar to manipulate circle constrains mid-session
*
* KNOWN ISSUES:
*   - Minor modification to circle detection algorithm is needed
'''

print __doc__

from imutils.video.pivideostream import PiVideoStream
from time import sleep
import cv2
import numpy as np

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
cv2.namedWindow("Live Feed")
cv2.setMouseCallback("Live Feed", control)

# Create a track bar for HoughCircles parameters
cv2.createTrackbar("dp", "Live Feed", 1, 50, placeholder)
cv2.createTrackbar("minDist", "Live Feed", 100, 500, placeholder)
cv2.createTrackbar("param1", "Live Feed", 55, 500, placeholder)
cv2.createTrackbar("param2", "Live Feed", 50, 500, placeholder)
cv2.createTrackbar("minRadius", "Live Feed", 50, 200, placeholder)
cv2.createTrackbar("maxRadius", "Live Feed", 100, 250, placeholder)

# Infinite loop
while True:
    
    # Get image from stream
    image = stream.read()
    
    # Convert into grayscale because HoughCircle only accepts grayscale images
    bgr2gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Get trackbar position and reflect it in HoughCircles parameters input
    dp = cv2.getTrackbarPos("dp", "Live Feed")
    dp = np.array(dp).astype(float)
    # Minimum distance between the centers of the detected circles
    # The smaller minDist is, the more neighbouring circles may be falsely detected.
    minDist = cv2.getTrackbarPos("minDist", "Live Feed")
    # The samller param2 is, the more false circles may be detected
    param1 = cv2.getTrackbarPos("param1", "Live Feed")
    param2 = cv2.getTrackbarPos("param2", "Live Feed")
    minRadius = cv2.getTrackbarPos("minRadius", "Live Feed")
    maxRadius = cv2.getTrackbarPos("maxRadius", "Live Feed")
    
    # Reduce Noise
    blur = cv2.medianBlur(bgr2gray,5)
    gaussBlur = cv2.GaussianBlur(blur,(5,5),0)

    # Adaptive threshold allows us to detect sharp edges in images
    threshold = cv2.adaptiveThreshold(gaussBlur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY,11,3.5)

    # Scan for circles
    circles = cv2.HoughCircles(threshold, cv2.HOUGH_GRADIENT, (dp/10), minDist,
                               param1, param2, minRadius, maxRadius)

    if circles is not None:
       for i in circles[0,:]:
           # Draw circle in green
           cv2.circle(image,(int(round(i[0])),int(round(i[1]))),int(round(i[2])),(0,255,0),5)
           cv2.circle(image,(int(round(i[0])),int(round(i[1]))),2,(0,255,0),10)

    # Live feed display toggle
    if normalDisplay:
        cv2.imshow("Live Feed", image)
        key = cv2.waitKey(1) & 0xFF
    elif not(normalDisplay):
        cv2.imshow("Live Feed", threshold)
        key = cv2.waitKey(1) & 0xFF
