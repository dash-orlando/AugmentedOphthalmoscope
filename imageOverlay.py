#!/usr/bin/python2.7
#A proof-of-concept image processing code
'''
* Code works and is capable of imposing one picture over another.
* Every process saves an image. That is used for testing purposes
* and is absolutely unnecessary.
* Known issues:
*   1- No fallback function in case no circles (retina) are detected. Code
*      simply terminates and exits.
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
img = Image('0.jpg')
img.show()

# Manipulate image for easier blob detection
invert = img.erode(1).invert()
invert.save('1.jpg')
invert.show()

# Find blobs and filter circles
blobs = invert.findBlobs()
if (blobs is not None):
    #isCircle(THRESHOLD). The higher THRESHOLD, the more circles are detected.
    circles = blobs.filter([b.isCircle(0.2) for b in blobs])
    blobs.draw()
    #Outline circles
    if circles:
        for b in circles:
            #uncomment line below to control radius filter
            #if int(b.radius()) > 1 and int(b.radius()) < 50:
            invert.drawCircle((b.x, b.y), b.radius(),SimpleCV.Color.BLUE,2)
            xy = (b.x, b.y)
            radius = int(b.radius())
            invert.show()
            invert.save('2.jpg')

# Load overlay
image = Image('2.jpg')
overlay = Image('overlay.jpg')

# subtract the greenscreen behind the overlay image
mask = overlay.hueDistance(color=Color.GREEN).binarize()
overlay = (overlay - mask)
overlaySize = overlay.size()

# Circular crop of the overlay in order to fit inside circle
overlay.drawCircle((overlaySize[0]/2, overlaySize[1]/2), radius, SimpleCV.Color.WHITE,10)
overlay = overlay.findBlobs()
if (overlay is not None):
    overlayCircles = blobs.filter([b.isCircle(0.2) for b in blobs])
    if overlayCircles:
        overlay = overlay[-1].crop()
        overlaySize = overlay.size()

#blit(image, coordinates=(x,y)). Where (x,y) is where the overlay image
#top left corner and the background image are pinned.
image.dl().blit(overlay, (xy[0]-(overlaySize[0]/2), xy[1]-(overlaySize[1]/2)))
image.show()
image.save('final.jpg')
