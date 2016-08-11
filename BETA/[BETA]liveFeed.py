'''
* Track a white circle using SimpleCV
* The parameters may need to be adjusted to match the RGB color the object.
* Known issues:
*       1- PiCamera is accessed as USB driver.
*       2- UV4L package displays a banner on the live display
*       3- Live display is blue-ish in color
*       4- Low framerate and lag on output
'''
print __doc__

import SimpleCV
from io import BytesIO
from time import sleep
from picamera import PiCamera
import cv2
import numpy
import scipy.misc

display = SimpleCV.Display(resolution = (640,480), title = "Live Feed ver0.1")
cam = PiCamera()
cam.resolution = (640, 480)
normaldisplay = True

while display.isNotDone():
    if display.mouseRight:
        normaldisplay = not(normaldisplay)
        print "Display Mode:", "Normal" if normaldisplay else "Segmented" 

    stream = BytesIO()
    cam.capture(stream, format='jpeg')
    buff = numpy.fromstring(stream.getvalue(), dtype=numpy.uint8)
    #img = cv2.imdecode(buff,0)
    img = scipy.misc.toimage(buff)
    #dist = img.colorDistance(SimpleCV.Color.BLACK).dilate(2)
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
