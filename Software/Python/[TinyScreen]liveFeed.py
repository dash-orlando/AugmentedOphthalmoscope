'''
* MODIFIED TO WORK ON SMALL TFT SCREEN
*
* NOTE: If overlay is NOT specified a sample overlay is chosen by default
* USEFUL ARGUMENTS:
*   -o/--overlay: Specify overlay file
*   -a/--alpha: Specify transperancy level (0.0 - 1.0)
*   -d/--debug: toggle to enable debugging mode (DEVELOPER ONLY!!!)
*
* VERSION: 0.9.5
*   - Added threads to distribute workload. Program now uses ~70% of
*     CPU power instead of ~30%
*   - Increased FPS from 7.5 to 35, a whooping 467% improvement!!!
*   - Added trackbars to change thresholding method and parameters mid-session
*
* KNOWN ISSUES:
*   - Overlay is lagging the output image (i.e circle is moved to
*     the left, overlay moves to the left half a second later)
*
* AUTHOR: Mohammad Odeh
* UPDATED: May 18th, 2017
* ----------------------------------------------------------
* ----------------------------------------------------------
*
* RIGHT CLICK: Shutdown Program.
* LEFT CLICK: Toggle view.
'''

ver = "Live Feed Ver0.9.5"
print __doc__

# Import necessary modules
import  numpy, cv2, argparse                                # Various Stuff
import  RPi.GPIO                    as      GPIO            # GPIO pins for peripherals (i.e LED)
from    imutils.video.pivideostream import  PiVideoStream   # Import threaded PiCam module
from    time                        import  sleep           # Sleep for stability
from    threading                   import  Thread          # Used to thread processes
from    Queue                       import  Queue           # Used to queue input/output
from    timeStamp                   import  fullStamp       # Show date/time on console output
from    imutils.video               import  FPS             # Benchmark FPS

# ************************************************************************
# =====================> CONSTRUCT ARGUMENT PARSER <=====================
# ************************************************************************
ap = argparse.ArgumentParser()

ap.add_argument("-o", "--overlay", required=False,
                help="path to overlay image")
ap.add_argument("-a", "--alpha", type=float, default=0.85,
                help="set alpha level (smaller = more transparent).\ndefault=0.85")
ap.add_argument("-d", "--debug", action='store_true',
                help="invoke flag to enable debugging")

args = vars( ap.parse_args() )

#args["debug"] = True
# ************************************************************************
# =====================> DEFINE NECESSARY FUNCTIONS <=====================
# ************************************************************************

# *************************************
# Define right/left mouse click events
# *************************************
def control( event, x, y, flags, param ):
    global normalDisplay
    
    # Right button shuts down program
    if event == cv2.EVENT_RBUTTONDOWN:
        # If debug flag is invoked
        if args["debug"]:
            fps.stop()
            print( fullStamp() + " [INFO] Elapsed time: {:.2f}".format(fps.elapsed()) )
            print( fullStamp() + " [INFO] Approx. FPS : {:.2f}".format(fps.fps()) )

        # Turn off LED
        GPIO.output( LED, GPIO.LOW )
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


# ****************************************************
# Define function to apply required filters to image
# ****************************************************
def procFrame(bgr2gray, Q_procFrame):

    # Dissolve noise while maintaining edge sharpness 
    bgr2gray = cv2.bilateralFilter( bgr2gray, 5, 17, 17 )
    bgr2gray = cv2.GaussianBlur(bgr2gray,(5,5),1)

    # Threshold any color that is not black to white
    retval, thresholded = cv2.threshold( bgr2gray, 455, 255, cv2.THRESH_TOZERO )

    kernel = cv2.getStructuringElement( cv2.MORPH_RECT, ( 10, 10 ) )
    bgr2gray = cv2.erode( cv2.dilate( thresholded, kernel, iterations=1 ), kernel, iterations=1 )

    # Place processed image in queue for retrieval
    Q_procFrame.put(bgr2gray)


# ******************************************************
# Define a function to scan for circles from camera feed
# ******************************************************

    # Error handling in case a non-allowable integer is chosen (1)
    try:
        # Scan for circles
        circles = cv2.HoughCircles( bgr2gray, cv2.HOUGH_GRADIENT, 8, 396,
                                    154, 99, 1, 14 )

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
                    overlay[ y1:y2, x1:x2 ] = resized

                    # Join overlay with live feed and apply specified transparency level
                    output = cv2.addWeighted( overlay, args["alpha"], frame, 1.0, 0 )
                    
                    # If debug flag is invoked
                    if args["debug"]:
                        # Draw circle
                        cv2.circle( output, ( x, y ), r, ( 0, 255, 0 ), 4 )
            
                # If not within window resolution keep looking
                else:
                    output = frame

                # Place output in queue for retrieval by main thread
                if Q_scan4circles.full() is False:
                    Q_scan4circles.put( [output, (x,y,r)] )

    # Error handling in case a non-allowable integer is chosen (2)
    except Exception as instance:
        print( fullStamp() + " Exception or Error Caught" )
        print( fullStamp() + " Error Type" + str(type(instance)) + "\n")
        #print( fullStamp() + " Error Arguments " + str( instance.arg ) + "\n" )
        print( fullStamp() + " Resetting ALL trackbars..." )

        # Exit function and re-loop
        return()

# ************************************************************************
# ===========================> SETUP PROGRAM <===========================
# ************************************************************************

# Setup GPIO pins and turn on LED
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
stream = PiVideoStream( resolution=(384, 288) ).start()
normalDisplay = True
sleep( 1.0 )

# Setup window and mouseCallback event
cv2.namedWindow( ver )
cv2.setMouseCallback( ver, control )

# Create a queue for retrieving data from thread
Q_procFrame = Queue( maxsize=0 )
Q_scan4circles = Queue( maxsize=0 )

# If debug flag is invoked
if args["debug"]:
    print( fullStamp() + " [INFO] Debug Mode: ON" )
    # Start benchmark
    fps = FPS().start()


# ************************************************************************
# =========================> MAKE IT ALL HAPPEN <=========================
# ************************************************************************

##initRun = True

# Infinite loop
while True:
    
    # Get image from stream
    frame = stream.read()[36:252, 48:336]
    output = frame

    # Add a 4th dimension (Alpha) to the captured frame
    (h, w) = frame.shape[:2]
    frame = numpy.dstack( [frame, numpy.ones( ( h, w ), dtype="uint8" ) * 255] )

    # Create an overlay layer
    overlay = numpy.zeros( ( h, w, 4 ), "uint8" )
    
    # Convert into grayscale because HoughCircle only accepts grayscale images
    bgr2gray = cv2.cvtColor( frame, cv2.COLOR_BGR2GRAY )

    # Start thread to process image and apply required filters to detect circles
    t_procFrame = Thread( target=procFrame, args=( bgr2gray, Q_procFrame ) )
    t_procFrame.daemon = True
    t_procFrame.start()

    # Check if queue has something available for retrieval
    if Q_procFrame.qsize() > 0:
        bgr2gray = Q_procFrame.get()
    
    # Start thread to scan for circles
    t_scan4circles = Thread( target=scan4circles, args=( bgr2gray, overlay, overlayImg, frame, Q_scan4circles ) )
    t_scan4circles.daemon = True
    t_scan4circles.start()

    # Check if queue has something available for retrieval
    if Q_scan4circles.qsize() > 0:
        output, (x_ROI, y_ROI, r_ROI) = Q_scan4circles.get()

    # If debug flag is invoked
    if args["debug"]:
       fps.update()

    # Live feed display toggle
    if normalDisplay:
        cv2.imshow(ver, output)
        key = cv2.waitKey(1) & 0xFF
    elif not(normalDisplay):
        cv2.imshow(ver, bgr2gray)
        key = cv2.waitKey(1) & 0xFF

# ************************************************************************
# =============================> DEPRECATED <=============================
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
