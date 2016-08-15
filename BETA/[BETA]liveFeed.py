'''
* Track a white circle using SimpleCV
* The parameters may need to be adjusted to match the RGB color the object.
* Known issues:
*       1- Super LOW framerate
*       2- width and height are switched for some reason
'''
print __doc__

from picamera.array import PiRGBArray
from picamera import PiCamera
import SimpleCV
import time
import cv2
import numpy

# Setup camera and display environment
display = SimpleCV.Display(resolution = (640,480), title = "Live Feed ver0.1")
camera = PiCamera()
camera.resolution = (640,480)
camera.framerate = 30
rawCapture = PiRGBArray(camera, size=(640,480))
normaldisplay = True
time.sleep(0.1)

while display.isNotDone():
    if display.mouseRight:
        normaldisplay = not(normaldisplay)
        print "Display Mode:", "Normal" if normaldisplay else "Segmented" 
    
    # Start capturing image stream
    for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        image = frame.array
        cv2RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = SimpleCV.Image(cv2RGB)
        dist = img.colorDistance(SimpleCV.Color.BLACK).dilate(2)
        segmented = img.stretch(200,255)
        blobs = segmented.findBlobs()
        dist = img.colorDistance(SimpleCV.Color.BLACK).dilate(2)
        segmented = img.stretch(200,255)
        blobs = segmented.findBlobs()
    if blobs:
        circles = blobs.filter([b.isCircle(0.2) for b in blobs])
        if circles:
            img.drawCircle((circles[-1].x, circles[-1].y), circles[-1].radius(),SimpleCV.Color.BLUE,3)
    cv2.imshow("test", image)
    key = cv2.waitKey(1) & 0xFF
    rawCapture.truncate(0)
    
    if normaldisplay:
        img.show()
    else:
        segmented.show()
