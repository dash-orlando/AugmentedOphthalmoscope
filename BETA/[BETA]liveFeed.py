'''
* Track a white circle using SimpleCV
* The parameters may need to be adjusted to match the RGB color the object.
* Known issues:
*       1- Super LOW framerate
*       2- width and height are switched for some reason
'''
print __doc__

import SimpleCV
import cv2
import numpy
from io import BytesIO
from time import sleep
from picamera import PiCamera

display = SimpleCV.Display(resolution = (640,480), title = "Live Feed ver0.1")
cam = PiCamera()
cam.resolution = (640, 480)
normaldisplay = True

while display.isNotDone():
    if display.mouseRight:
        normaldisplay = not(normaldisplay)
        print "Display Mode:", "Normal" if normaldisplay else "Segmented" 
    
    stream = BytesIO()
    #give time to initialize
    sleep(2)
    cam.capture(stream, format='jpeg')
    buff = numpy.fromstring(stream.getvalue(), dtype=numpy.uint8)
    cv2img = cv2.imdecode(buff,1)
    cv2RGB = cv2.cvtColor(cv2img, cv2.COLOR_BGR2RGB)
    img = SimpleCV.Image(cv2RGB)
    dist = img.colorDistance(SimpleCV.Color.BLACK).dilate(2)
    segmented = img.stretch(200,255)
    blobs = segmented.findBlobs()
    if blobs:
        circles = blobs.filter([b.isCircle(0.2) for b in blobs])
        if circles:
            img.drawCircle((circles[-1].x, circles[-1].y), circles[-1].radius(),SimpleCV.Color.BLUE,3)

    if normaldisplay:
        img.show()
    else:
        segmented.show()
