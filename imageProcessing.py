#!/usr/bin/python2.7
#A proof-of-concept image processing code

from picamera import PiCamera
from SimpleCV import Image
import time

camera = PiCamera()
camera.resolution = (640, 480)
camera.start_preview()
time.sleep(5)
camera.capture('image.jpg')
img = Image("image.jpg")

eyes = img.findHaarFeatures('two_eyes_big.xml')
    
if ( eyes is not None):
    eyes = eyes[-1].crop()
    eyes.show()
else:
    print "Not found"

camera.stop_preview()
img.show()
zoomEyes = eyes.scale(3)
zoomEyes.save('eyes.jpg')
zoomEyes.show()
time.sleep(5)
