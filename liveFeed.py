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

display = SimpleCV.Display(resolution = (640,480), title = "Live Feed ver0.1")
cam = SimpleCV.Camera()
cam.resolution = (640, 480)
normaldisplay = True

# Initiate Display
while display.isNotDone():
    if display.mouseRight:
        normaldisplay = not(normaldisplay)
        print "Display Mode:", "Normal" if normaldisplay else "Segmented" 

    # Start feed
    img = cam.getImage().flipHorizontal()
    # Modify image for easier blob detection
    dist = img.colorDistance(SimpleCV.Color.BLACK).dilate(2)
    segmented = dist.stretch(200,255)
    # If blobs are detected find circles
    blobs = segmented.findBlobs()
    if blobs:
        # Filter circles
        circles = blobs.filter([b.isCircle(0.2) for b in blobs])
        if circles:
            img.drawCircle((circles[-1].x, circles[-1].y), circles[-1].radius(),SimpleCV.Color.BLUE,3)

    if normaldisplay:
        img.show()
    else:
        segmented.show()
