'''
* NOTE: If overlay is NOT specified a sample overlay is chosen by default
* USEFUL ARGUMENTS:
*   -o/--overlay: Specify overlay file
*   -a/--alpha: Specify transperancy level (0.0 - 1.0)
*   -d/--debug: toggle to enable debugging mode (DEVELOPER ONLY)
*
* VERSION: 0.9.3
*   - Changed a couple of parameters that improved detection accuracy
*
* KNOWN ISSUES:
*   - Non atm!!
*
* AUTHOR: Mohammad Odeh
* UPDATED: May 8th, 2017
* ----------------------------------------------------------
* ----------------------------------------------------------
*
* RIGHT CLICK: Shutdown Program.
* LEFT CLICK: Toggle view.
'''

ver = "Live Feed Ver0.9.3"
print __doc__

# Import necessary modules
import numpy, cv2, argparse
import RPi.GPIO                     as GPIO
from imutils.video.pivideostream    import PiVideoStream
from time                           import sleep
#from LEDRing                        import *

#args["debug"] = 1


# ************************************************************************
# CONSTRUCT ARGUMENT PARSER
# ************************************************************************
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--overlay", required=False,
                help="path to overlay image")
ap.add_argument("-a", "--alpha", type=float, default=0.85,
                help="set alpha level (smaller = more transparent).\ndefault=0.85")
ap.add_argument("-d", "--debug", action='store_true',
                help="invoke flag to enable debugging")

##Argument Parser for LED ring brightness (deprecated for the time being)
##ap.add_argument("-b", "--brightness", type=int, default=50,
##                help="set brightness level")

args = vars(ap.parse_args())


# ************************************************************************
# DEFINE NECESSARY FUNCTIONS
# ************************************************************************

# *************************************
# Define right/left mouse click events
# *************************************
def control( event, x, y, flags, param ):
    global normalDisplay
    
    # Right button shuts down program
    if event == cv2.EVENT_RBUTTONDOWN:
        # Turn off LED
##        colorWipe( strip, Color( 0, 0, 0, 0 ), 0 )
        GPIO.output(LED,GPIO.LOW)
        stream.stop()
        cv2.destroyAllWindows()
        quit()
        
    # Left button toggles display
    elif event == cv2.EVENT_LBUTTONDOWN:
        normalDisplay=not( normalDisplay )


# ****************************************************
# Define a placeholder function for trackbar. This is
# needed for the trackbars to function properly.
# ****************************************************
def placeholder( x ):
    pass


# ************************************************************************
# SETUP PROGRAM
# ************************************************************************

LED = 21
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
GPIO.setup(LED,GPIO.OUT)
GPIO.output(LED,GPIO.HIGH)

# Check whether an overlay is specified
if args["overlay"] is not None:
    overlayImg = cv2.imread( args["overlay"], cv2.IMREAD_UNCHANGED )
else:
    overlayImg = cv2.imread( "Overlay.png", cv2.IMREAD_UNCHANGED )

# Load overlay image with Alpha channel
( wH, wW ) = overlayImg.shape[:2]
( B, G, R, A ) = cv2.split( overlayImg )
B = cv2.bitwise_and( B, B, mask=A )
G = cv2.bitwise_and( G, G, mask=A )
R = cv2.bitwise_and( R, R, mask=A )
overlayImg = cv2.merge( [B, G, R, A] )

# Setup camera
stream = PiVideoStream( hf=True ).start()
normalDisplay = True
sleep( 1.0 )

### Turn on LED Ring
##colorWipe(strip, Color(255, 255, 255, 255), 0)
##colorWipe(strip, Color(0, 0, 0, 0), 0)

# Setup window and mouseCallback event
cv2.namedWindow( ver )
cv2.setMouseCallback( ver, control )

# Create a track bar for HoughCircles parameters
cv2.createTrackbar( "dp", ver, 9, 50, placeholder )
cv2.createTrackbar( "param2", ver, 43, 750, placeholder )
cv2.createTrackbar( "minRadius", ver, 10, 200, placeholder )
cv2.createTrackbar( "maxRadius", ver, 30, 250, placeholder )


# ************************************************************************
# MAKE IT ALL HAPPEN
# ************************************************************************

# Infinite loop
while True:
    
    # Get image from stream
    frame = stream.read()[46:322, 60:420]
    output = frame
    
    # Add a 4th dimension (Alpha) to the captured frame
    (h, w) = frame.shape[:2]
    frame = numpy.dstack( [frame, numpy.ones( ( h, w ), dtype="uint8" ) * 255] )

    # Create an overlay layer
    overlay = numpy.zeros( ( h, w, 4 ), "uint8" )
    
    # Convert into grayscale because HoughCircle only accepts grayscale images
    bgr2gray = cv2.cvtColor( frame, cv2.COLOR_BGR2GRAY )
    bgr2gray = cv2.bilateralFilter( bgr2gray, 11, 17, 17 )

    # Threshold any color that is not black to white
    retval, thresholded = cv2.threshold( bgr2gray, 60, 158, cv2.THRESH_BINARY )

    kernel = cv2.getStructuringElement( cv2.MORPH_RECT, ( 10, 10 ) )
    bgr2gray = cv2.erode( cv2.dilate( thresholded, kernel, iterations=1 ), kernel, iterations=1 )

    
    # Get trackbar position and reflect it in HoughCircles parameters input
    dp = cv2.getTrackbarPos( "dp", ver )
    param2 = cv2.getTrackbarPos( "param2", ver )
    minRadius = cv2.getTrackbarPos( "minRadius", ver )
    maxRadius = cv2.getTrackbarPos( "maxRadius", ver )
    
    
    # Scan for circles
    circles = cv2.HoughCircles( bgr2gray, cv2.HOUGH_GRADIENT, dp, 396,
                                191, param2, minRadius, maxRadius )

    '''
    Experimental values:            Original Values:
    dp = 9                          dp = 9
    minDist = 396                   minDist = 396
    param1 = 191                    param1 = 191
    param2 = 43                     param2 = 43
    minRadius = 10                  minRadius = 1
    maxRadius = 30                  maxRadius = 16
    '''

    # If circles are found draw them
    if circles is not None:
        circles = numpy.round( circles[0,:] ).astype( "int" )
        for ( x, y, r ) in circles:

            # Resize watermark image
            resized = cv2.resize( overlayImg, ( 2*r, 2*r ),
                                  interpolation = cv2.INTER_AREA )

            # Retrieve overlay location
            y1 = y-r
            y2 = y+r
            x1 = x-r
            x2 = x+r

            # Check whether overlay location is within window resolution
            if x1>0 and x1<w and x2>0 and x2<w and y1>0 and y1<h and y2>0 and y2<h:
                # Place overlay image inside circle
                overlay[y1:y2,x1:x2]=resized

                # Join overlay with live feed and apply specified transparency level
                output = cv2.addWeighted( overlay, args["alpha"], frame, 1.0, 0 )

                # If debug flag is invoked
                if args["debug"] is True:
                    # Draw circle
                    cv2.circle( output, ( x, y ), r, ( 0, 255, 0 ), 4 )
##                    cv2.putText( output, "%.2fmm" % ( mm ),
##                                 ( output.shape[1] - 150, output.shape[0] - 15 ),
##                                 cv2.FONT_HERSHEY_SIMPLEX, 1.0, ( 0, 255, 0 ), 3 )
        
            # If not within window resolution keep looking
            else:
                output = frame
    
    # Live feed display toggle
    if normalDisplay:
        cv2.imshow(ver, output)
        cv2.imshow( "calibrationTool", bgr2gray )
        key = cv2.waitKey(1) & 0xFF
    elif not(normalDisplay):
        cv2.imshow(ver, bgr2gray)
        key = cv2.waitKey(1) & 0xFF



# ************************************************************************
# DEPRECATED
# ************************************************************************

'''
bgr2gray = cv2.erode(cv2.dilate(thresholded, kernel, iterations=1), kernel, iterations=1)

# Convert into HSV and threshold black
lower = numpy.array([0,0,0])
upper = numpy.array([0,0,25])
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv, lower, upper)

# Get trackbar position and reflect it in HoughCircles parameters input

-----------------------------------------------------------------------
maxRadius = cv2.getTrackbarPos("maxRadius", ver)

# Reduce Noise
blur = cv2.medianBlur(bgr2gray,5)
gaussBlur = cv2.GaussianBlur(blur,(5,5),0)

# Adaptive threshold allows us to detect sharp edges in images
threshold = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY,11,3.5)

# Scan for circles

------------------------------------------------------------------------

# ****************************************************
# Define function to calibrate camera by extracting
# dimensions from a known image with known dimensions
# ****************************************************
def find_marker( image ):
    # Convert into grayscale because HoughCircle only accepts grayscale images
    bgr2gray = cv2.cvtColor( image, cv2.COLOR_BGR2GRAY )
    bgr2gray = cv2.bilateralFilter( bgr2gray,11,17,17 )

    # Threshold any color that is not black to white
    retval, thresholded = cv2.threshold(bgr2gray, 180, 255, cv2.THRESH_BINARY )

    kernel = cv2.getStructuringElement( cv2.MORPH_RECT, ( 10, 10 ) )
    bgr2gray = cv2.erode( cv2.dilate( thresholded, kernel, iterations=1 ), kernel, iterations=1 )

    cv2.imshow( "calibrationTool", bgr2gray )
    
    # Find (in future update, the largest) circle outline
    circles = cv2.HoughCircles( bgr2gray, cv2.HOUGH_GRADIENT, 14, 396,
                                191, 43, 50, 85 )

    if circles is not None:
        circles = numpy.round( circles[0,:] ).astype( "int" )
        if args["debug"] is True:
            print( "Shape: " + str( circles.shape ) )
            print( "Array content: " + str( circles ) )
        for ( x, y, r ) in circles:
            circle = ( x, y, r )
            return( x, y, r )

    else:
        return(0)
        

# ************************************************************************
# Define function that returns distance from object to camera
# ************************************************************************
def distance_to_camera( knownWidth, focalLength, perWidth ):
    # Compute and return the distance from the object to the camera
    return ( knownWidth * focalLength ) / perWidth

--------------------------------------------------------------------------

cv2.createTrackbar( "maxRadius", ver, 30, 250, placeholder )

# ************************************************************************
# TEMPORARELY DEPRECATED
# ************************************************************************
### Calibrate camera using a predefined scale for distance detection
##KNOWN_DISTANCE = 3.5
##KNOWN_WIDTH = 2
##image = cv2.imread( "images/3.5inch.png" )
##image = cv2.resize( image, ( 360, 276 ) )
##marker = find_marker( image )
##focalLength = ( marker[2] * KNOWN_DISTANCE ) / KNOWN_WIDTH

#KNOWN_WIDTH = 0.4645669 #Average iris diameter

# ************************************************************************
# MAKE IT ALL HAPPEN
# ************************************************************************

--------------------------------------------------------------------------

for ( x, y, r ) in circles:

            # ************************************************************************
            # TEMPORARELY DEPRECATED
            # ************************************************************************
            
##            # Get distance away from camera
##            marker = find_marker( output.copy() )
##            
##            if marker is not 0:
##                KNOWN_WIDTH = (2*r)/96.0
##                if args["debug"] is True:
##                    print( "Detected Width: %.2f" %KNOWN_WIDTH )
##                inches = distance_to_camera( KNOWN_WIDTH, focalLength, marker[2] )
##                mm = inches*25.4 #Get distance in millimeters

            # Resize watermark image
            resized = cv2.resize( overlayImg, ( 2*r, 2*r ),
                                  interpolation = cv2.INTER_AREA )

'''
