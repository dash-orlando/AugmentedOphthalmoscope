'''
* Track a white circle using SimpleCV
* The parameters may need to be adjusted to match the RGB color the object.
* Known issues:
*       1- PiCamera is accessed as USB driver.
*       2- UV4L package displays a banner on the live display
*       3- Live display is blue-ish in color
'''
print __doc__

import SimpleCV

display = SimpleCV.Display()
cam = SimpleCV.Camera()
normaldisplay = True

while display.isNotDone():

	if display.mouseRight:
		normaldisplay = not(normaldisplay)
		print "Display Mode:", "Normal" if normaldisplay else "Segmented" 
	
	img = cam.getImage().flipHorizontal()
	dist = img.colorDistance(SimpleCV.Color.BLACK).dilate(2)
	segmented = dist.stretch(200,255)
	blobs = segmented.findBlobs()
	if blobs:
		circles = blobs.filter([b.isCircle(0.5) for b in blobs])
		if circles:
			img.drawCircle((circles[-1].x, circles[-1].y), circles[-1].radius(),SimpleCV.Color.BLUE,3)

	if normaldisplay:
		img.show()
	else:
		segmented.show()
