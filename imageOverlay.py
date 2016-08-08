#!/usr/bin/python2.7
#A proof-of-concept image processing code
'''
Code works and is capable of imposing one picture over another.
Known issues:
    1- For some reason I am having trouble inserting the image layer onto the
       xy-coordinate I want.
    2- No fallback function in case no circles (retina) was detected. Code
       simply terminates and exits.
'''

from picamera import PiCamera
from SimpleCV import *
import time

# Initiate picamera as camera and set resolution
camera = PiCamera()
camera.resolution = (1024, 768)
camera.start_preview()
time.sleep(3)

# Capture Image
camera.capture('0.jpg')
img = Image("0.jpg")
camera.stop_preview()
img.show()

# Manipulate image for easier blob detection
invert = img.erode(2).invert()
invert.save('1.jpg')
invert.show()

# Find blobs and filter circles
blobs = invert.findBlobs()
if (blobs is not None):
    circles = blobs.filter([b.isCircle(0.1) for b in blobs])
    blobs.draw()
    if circles:
        for b in circles:
            #if int(b.radius()) > 1 and int(b.radius()) < 50:
            invert.drawCircle((b.x, b.y), b.radius(),SimpleCV.Color.BLUE,2)
            xy = (b.x, b.y)
            radius = int(b.radius())
            invert.show()
            invert.save('2.jpg')

# Load overlay
image = Image('2.jpg')
overlay = Image('hqdefault.jpg')
image.dl().blit(overlay, xy)
image.show()
'''overlaydl = DrawingLayer((radius*2,radius*2))
overlaydl.blit(overlay, xy)
image.addDrawingLayer(overlaydl)
image.applyLayers()
image.show()
'''
