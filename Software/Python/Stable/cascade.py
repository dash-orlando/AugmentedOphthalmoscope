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

##            ToF.close()                     # Close port
##            if ( t_getDist.isAlive() ):
##                t_getDist.join(5.0)         # Terminate serial port pooling thread
##                if args["debug"]:           # If debug flag is invoked, display message
##                    print( fullStamp() + " getDist: Terminated" )

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


# ************************************************************************
# ===========================> SETUP PROGRAM <===========================
# ************************************************************************

# Setup camera
stream = PiVideoStream( resolution=(384, 288) ).start()
normalDisplay = True
sleep( 1.0 )

# Setup window and mouseCallback event
cv2.namedWindow( ver )
cv2.setMouseCallback( ver, control )

# Setup window and trackbars for AI view
cv2.namedWindow( "AI_View" )

# Load cascade classifier
detector = cv2.CascadeClassifier('cascade.xml')

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
    grey = cv2.cvtColor( output, cv2.COLOR_BGR2GRAY )

    # run detector
    eyes = detector.detectMultiScale( grey, scaleFactor=1.3,
                                       minNeighbors=10, minSize=(45, 45) )

    # Draw if we detect something
    if( eyes is not None ):
        for( x, y, w, h ) in eyes:
            cv2.rectangle( output, (x,y), (x+w, y+h), (255,255,0), 2 )
    else: pass

    # If debug flag is invoked
    if args["debug"]:
       fps.update()

    # Live feed display toggle
    if normalDisplay:
        cv2.imshow(ver, output)
        cv2.imshow( "AI_View", grey )
        key = cv2.waitKey(1) & 0xFF
    elif not(normalDisplay):
        cv2.imshow(ver, grey)
        key = cv2.waitKey(1) & 0xFF


