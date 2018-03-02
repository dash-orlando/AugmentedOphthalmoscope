'''
* NOTE: If overlay is NOT specified a sample overlay is chosen by default
* USEFUL ARGUMENTS:
*   -o/--overlay: Specify overlay file
*   -a/--alpha: Specify transperancy level (0.0 - 1.0)
*   -d/--debug: toggle to enable debugging mode (DEVELOPER ONLY!!!)
*
* VERSION: 0.9.6
*   - Threads now safely exit at program shutdown
*   - FPS dropped from ~35 to ~28 due to the addition of constant
*     serial checking/pooling.
*   - Incorporated ToF sensor: overlay is now triggered ONLY at a
*     preset distance away from target.
*   - Reduced "jittering" of overlay by manipulating parameters
*
* KNOWN ISSUES:
*   - pyserial & threading do NOT play nicely and conflicts arise.
*     An IOError is raised at program shutdown, nonetheless, this
*     does NOT affect program operation. Error can be safely ignored.
*
* AUTHOR:   Mohammad Odeh
* WRITTEN:  Aug  1st, 2016
* UPDATED:  Jul 11th, 2017
* ----------------------------------------------------------
* ----------------------------------------------------------
*
* RIGHT CLICK: Shutdown Program.
* LEFT CLICK: Toggle view.
'''

ver = "Live Feed Ver0.9.6"
print __doc__

# Import necessary modules
import  numpy, cv2, argparse                                # Various Stuff
from    imutils.video.pivideostream import  PiVideoStream   # Import threaded PiCam module
from    imutils.video               import  FPS             # Benchmark FPS
from    time                        import  sleep           # Sleep for stability
from    threading                   import  Thread          # Used to thread processes
from    Queue                       import  Queue           # Used to queue input/output
from    timeStamp                   import  fullStamp       # Show date/time on console output
from    usbProtocol                 import  createUSBPort   # Create USB Port

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

args["debug"] = True
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

        # Do some shutdown clean up
        try:
            if ( t_scan4circles.isAlive() ):
                t_scan4circles.join(5.0)    # Terminate circle scanning thread
                if args["debug"]:           # If debug flag is invoked, display message
                    print( fullStamp() + " scan4circles: Terminated" )
                
            if ( t_procFrame.isAlive() ):
                t_procFrame.join(5.0)       # Terminate image processing thread
                if args["debug"]:           # If debug flag is invoked, display message
                    print( fullStamp() + " procFrame: Terminated" )


        except Exception as e:
            print( "Caught Error: %s" %str( type(e) ) )

        finally:
            stream.stop()                   # Stop capturing frames from stream
            cv2.destroyAllWindows()         # Close any open windows
            quit()                          # Shutdown python interpreter
        
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

    # Get trackbar position and reflect it threshold type and values
    threshType = cv2.getTrackbarPos( "Type:\n0.Binary\n1.BinaryInv\n2.Trunc\n3.2_0\n4.2_0Inv",
                                     "AI_View")
    thresholdVal    = cv2.getTrackbarPos( "thresholdVal", "AI_View")
    maxValue        = cv2.getTrackbarPos( "maxValue"    , "AI_View")

    # Dissolve noise while maintaining edge sharpness 
    bgr2gray = cv2.bilateralFilter( bgr2gray, 5, 17, 17 ) #( bgr2gray, 11, 17, 17 )
    bgr2gray = cv2.GaussianBlur( bgr2gray, (5, 5), 1 )

    # Threshold any color that is not black to white
    if threshType == 0:
        retval, thresholded = cv2.threshold( bgr2gray, thresholdVal, maxValue, cv2.THRESH_BINARY )
    elif threshType == 1:
        retval, thresholded = cv2.threshold( bgr2gray, thresholdVal, maxValue, cv2.THRESH_BINARY_INV )
    elif threshType == 2:
        retval, thresholded = cv2.threshold( bgr2gray, thresholdVal, maxValue, cv2.THRESH_TRUNC )
    elif threshType == 3:
        retval, thresholded = cv2.threshold( bgr2gray, thresholdVal, maxValue, cv2.THRESH_TOZERO )
    elif threshType == 4:
        retval, thresholded = cv2.threshold( bgr2gray, thresholdVal, maxValue, cv2.THRESH_TOZERO_INV )

    kernel = cv2.getStructuringElement( cv2.MORPH_RECT, ( 10, 10 ) )
    bgr2gray = cv2.erode( cv2.dilate( thresholded, kernel, iterations=1 ), kernel, iterations=1 )

    # Place processed image in queue for retrieval
    Q_procFrame.put(bgr2gray)


# ******************************************************
# Define a function to scan for circles from camera feed
# ******************************************************
def scan4circles( bgr2gray, overlay, overlayImg, frame, Q_scan4circles ):

    # Scan for circles
    circles = cv2.HoughCircles( bgr2gray, cv2.HOUGH_GRADIENT, dp, minDist,
                                param1, param2, minRadius, maxRadius )


    # If circles are found draw them
    if circles is not None:
        
        # Extract the image we want!
        circle1 = numpy.uint16( numpy.around(circles) )
        mask = numpy.full( (frame.shape[0], frame.shape[1]), 0, dtype=numpy.uint8 )

        for i in circle1[0,:] :
            cv2.circle( mask, (i[0], i[1]), i[2], (255,255,255), -1)

        fg = cv2.bitwise_or( frame, frame, mask=mask )
        
        mask = cv2.bitwise_not( mask )
        bg = numpy.full( frame.shape, 255, dtype=numpy.uint8 )
        bk = cv2.bitwise_or( bg, bg, mask=mask )

        final = cv2.bitwise_or(fg, bk)
        

        # Place output in queue for retrieval by main thread
        if Q_scan4circles.full() is False:
            Q_scan4circles.put( final )

    else:
        # Place output in queue for retrieval by main thread
        if Q_scan4circles.full() is False:
            Q_scan4circles.put( frame )


# ************************************************************************
# ===========================> SETUP PROGRAM <===========================
# ************************************************************************

# Check whether an overlay is specified
if args["overlay"] is not None:
    overlayImg = cv2.imread( args["overlay"], cv2.IMREAD_UNCHANGED )
else:
    overlayImg = cv2.imread( "Overlay.png"  , cv2.IMREAD_UNCHANGED )

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

# Create a track bar for HoughCircles parameters
cv2.createTrackbar( "dp"        , ver, 34   , 50 , placeholder ) #14
cv2.createTrackbar( "minDist"   , ver, 396  , 750, placeholder )
cv2.createTrackbar( "param1"    , ver, 316  , 750, placeholder ) #326
cv2.createTrackbar( "param2"    , ver, 236  , 750, placeholder ) #231
cv2.createTrackbar( "minRadius" , ver, 7    , 200, placeholder ) #1
cv2.createTrackbar( "maxRadius" , ver, 14   , 250, placeholder )

# Setup window and trackbars for AI view
cv2.namedWindow( "AI_View" )

cv2.createTrackbar( "Type:\n0.Binary\n1.BinaryInv\n2.Trunc\n3.2_0\n4.2_0Inv",
                    "AI_View", 3, 4, placeholder )
cv2.createTrackbar( "thresholdVal", "AI_View", 30, 254, placeholder ) #45
cv2.createTrackbar( "maxValue", "AI_View", 255, 255, placeholder )

# Create a queue for retrieving data from thread
Q_procFrame     = Queue( maxsize=0 )
Q_scan4circles  = Queue( maxsize=0 )

# If debug flag is invoked
if args["debug"]:
    # Start benchmark
    print( fullStamp() + " [INFO] Debug Mode: ON" )
    fps = FPS().start()


# ************************************************************************
# =========================> MAKE IT ALL HAPPEN <=========================
# ************************************************************************

# Infinite loop
while True:
    
    # Get image from stream
    frame = stream.read()[36:252, 48:336]
    output = frame

    # Convert into grayscale because HoughCircle only accepts grayscale images
    bgr2gray = cv2.cvtColor( frame, cv2.COLOR_BGR2GRAY )

    # Start thread to process image and apply required filters to detect circles
    t_procFrame = Thread( target=procFrame, args=( bgr2gray, Q_procFrame ) )
    t_procFrame.start()

    # Check if queue has something available for retrieval
    if Q_procFrame.qsize() > 0:
        bgr2gray = Q_procFrame.get()

    # Get trackbar position and reflect it in HoughCircles parameters input
    dp = cv2.getTrackbarPos( "dp", ver )
    minDist = cv2.getTrackbarPos( "minDist", ver )
    param1 = cv2.getTrackbarPos( "param1", ver )
    param2 = cv2.getTrackbarPos( "param2", ver )
    minRadius = cv2.getTrackbarPos( "minRadius", ver )
    maxRadius = cv2.getTrackbarPos( "maxRadius", ver )

    # Start thread to scan for circles
    t_scan4circles = Thread( target=scan4circles, args=( bgr2gray, overlay, overlayImg, frame, Q_scan4circles ) )
    t_scan4circles.start()

    # Check if queue has something available for retrieval
    if Q_scan4circles.qsize() > 0:
        output = Q_scan4circles.get()

    # If debug flag is invoked
    if args["debug"]:
       fps.update()

    # Live feed display toggle
    if normalDisplay:
        cv2.imshow(ver, output)
        cv2.imshow( "AI_View", bgr2gray )
        key = cv2.waitKey(1) & 0xFF
    elif not(normalDisplay):
        cv2.imshow(ver, bgr2gray)
        key = cv2.waitKey(1) & 0xFF
