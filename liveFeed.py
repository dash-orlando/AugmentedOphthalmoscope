'''
* NOTE: Place Overlay.png in same directory as liveFeed.py
*
* VERSION: 0.7
*   - Program can successfully detect a circle (pupil) at
*     close range (~35mm).
*   - Resolution reduced to (480, 368).
*   - Added min/maxRadius trackbars to manipulate circle
*     constraints mid-session.
*   - Overlaying an image inside the detected circle (pupil)
*     is now possible.
*
* KNOWN ISSUES:
*   - Minor modification to circle detection algorithm is needed.
*   - Program crashed when no circles are detected.
*
* AUTHOR: Mohammad Odeh
*
* ----------------------------------------------------------
*                           Instructions
* ----------------------------------------------------------
* 1) Place camera about 35mm away from target and make sure
*    target is well illuminated.
* 2) MAGIC!!
*
* RIGHT CLICK: Shutdown Program.
* LEFT CLICK: Toggle view.
'''

print __doc__

# Import necessary modules
from imutils.video.pivideostream import PiVideoStream
from time import sleep
import imutils
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

# Load overlay image with Alpha channel
overlayImg = cv2.imread("Overlay.png", cv2.IMREAD_UNCHANGED)
(wH, wW) = overlayImg.shape[:2]
(B, G, R, A) = cv2.split(overlayImg)
B = cv2.bitwise_and(B, B, mask=A)
G = cv2.bitwise_and(G, G, mask=A)
R = cv2.bitwise_and(R, R, mask=A)
overlayImg = cv2.merge([B, G, R, A])

# Setup camera
stream = PiVideoStream(hf=True).start()
normalDisplay = True
sleep(2)

# Setup window and mouseCallback event
cv2.namedWindow("Live Feed ver0.7")
cv2.setMouseCallback("Live Feed ver0.7", control)

# Create a track bar for HoughCircles parameters
cv2.createTrackbar("minRadius", "Live Feed ver0.7", 5, 200, placeholder)
cv2.createTrackbar("maxRadius", "Live Feed ver0.7", 70, 250, placeholder)

# Infinite loop
while True:
    
    # Get image from stream
    frame = stream.read()
    
    # Add a 4th dimension (Alpha) to the captured frame
    (h, w) = frame.shape[:2]
    frame = numpy.dstack([frame, numpy.ones((h, w), dtype="uint8") * 255])

    # Create an overlay layer
    overlay = numpy.zeros((h,w,4), "uint8")
    
    # Convert into grayscale because HoughCircle only accepts grayscale images
    bgr2gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Get trackbar position and reflect it in HoughCircles parameters input
    minRadius = cv2.getTrackbarPos("minRadius", "Live Feed ver0.7")
    maxRadius = cv2.getTrackbarPos("maxRadius", "Live Feed ver0.7")
    
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
    Experimental values:            Original Values (buggy):
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
            if (x,y,r) is not None:
                # Resize watermark image
                resized = cv2.resize(overlayImg, (2*r, 2*r),
                                     interpolation = cv2.INTER_AREA)

                # Place overlay image inside circle
                overlay[y-r:y+r,x-r:x+r]=resized
    
                # Join overlay and live feed
                output = numpy.array(numpy.clip(frame+overlay, 0,255),"uint8")
                 
            # Draw circle
            cv2.circle(output, (x,y),r,(0,255,0),4)
    
    # Live feed display toggle
    if normalDisplay:
        cv2.imshow("Live Feed ver0.7", output)
        key = cv2.waitKey(1) & 0xFF
    elif not(normalDisplay):
        cv2.imshow("Live Feed ver0.7", threshold)
        key = cv2.waitKey(1) & 0xFF
