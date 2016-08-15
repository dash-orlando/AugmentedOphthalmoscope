'''
* VERSION: 0.3
*   - Track any circular object instead of a specific color
*   - Huge improvements to the framerate
*   - Dropped the need for extra drivers and now runs picam module natively
*
* KNOWN ISSUES:
*       1- Circle detection needs a lot of improvement and filtering
*       2- Framerate still needs improvement
'''

print __doc__

from picamera.array import PiRGBArray
from picamera import PiCamera
import SimpleCV
import time
import cv2
import numpy

# Setup camera and display environment
display = cv2.namedWindow("Live Feed ver0.2")
camera = PiCamera()
camera.resolution = (640,480)
camera.framerate = 30
rawCapture = PiRGBArray(camera, size=(640,480))
normaldisplay = True
time.sleep(0.1)

# Start capturing image stream
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    image = frame.array
    #Reduce Noise
    image = cv2.medianBlur(image,5)
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(grayscale, cv2.cv.CV_HOUGH_GRADIENT,1.2,100,
                               param1=55,param2=50,minRadius=1,maxRadius=50)

    if circles is not None:
        circles = numpy.round(circles[0,:]).astype("int")
        for (x, y, r) in circles:
            # Draw circle
            cv2.circle(image, (x,y),r,(0,255,0),4)

    cv2.imshow("test", image)
    key = cv2.waitKey(1) & 0xFF
    rawCapture.truncate(0)
'''
    #ORIGINAL BEGIN
    img = SimpleCV.Image(cv2RGB)
    #dist = img.colorDistance(SimpleCV.Color.BLACK).dilate(2)
    #segmented = img.stretch(200,255)
    #blobs = segmented.findBlobs()
    #ORIGINAL END
    dist = img.colorDistance(SimpleCV.Color.BLACK).dilate(2)
    segmented = img.stretch(200,255)
    blobs = segmented.findBlobs()
    if blobs:
        circles = blobs.filter([b.isCircle(0.2) for b in blobs])
        if circles:
            for b in circles:
                if int(b.radius()) > 1 and int(b.radius()) < 50:
                    cv2.circle(image, (circles[-1].x, circles[-1].y), circles[-1].radius(),SimpleCV.Color.BLUE,3)
    cv2.imshow("test", image)
# BEGIN ''
    if normaldisplay:
        cv2.imshow("test", img)
    else:
        segmented.show()
# END ''
'''
