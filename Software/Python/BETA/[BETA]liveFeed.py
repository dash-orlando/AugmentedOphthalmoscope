'''
* NOTE: If overlay is NOT specified a sample overlay is chosen by default
* USEFUL ARGUMENTS:
*   -o/--overlay: Specify overlay file
*   -a/--alpha: Specify transperancy level (0.0 - 1.0)
*   -d/--debug: toggle to enable debugging mode (DEVELOPER ONLY!!!)
*
* VERSION: 0.9.7
*   - Uses adaptive thresholding instead of global thresholding
*   - Split RGB channels and perform preprocessing on them seperately 
*
* KNOWN ISSUES:
*   - Light colored eyes are almost impossible to detect
*
* AUTHOR:   Mohammad Odeh
* WRITTEN:  Aug  1st, 2016
* UPDATED:  Jul 26th, 2017
* ----------------------------------------------------------
* ----------------------------------------------------------
*
* RIGHT CLICK: Shutdown Program.
* LEFT CLICK: Toggle view.
'''

ver = "Live Feed Ver0.9.7"
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

            ToF.close()                     # Close port
            if ( t_getDist.isAlive() ):
                t_getDist.join(5.0)         # Terminate serial port pooling thread
                if args["debug"]:           # If debug flag is invoked, display message
                    print( fullStamp() + " getDist: Terminated" )

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
    threshType = cv2.getTrackbarPos( "Type:\n0.MEAN_Binary\n1.GAUSSIAN_Binary\n2.MEAN_BinaryInv\n3.GAUSSIAN_BinaryInv\n4.OTSU",
                                     "AI_View")

    maxValue        = cv2.getTrackbarPos( "maxValue"    , "AI_View")
    blockSize       = cv2.getTrackbarPos( "blockSize"   , "AI_View")
    cte             = cv2.getTrackbarPos( "cte"         , "AI_View")
    GaussianBlur    = cv2.getTrackbarPos( "GaussianBlur", "AI_View")

    # Make sure blockSize is an odd number
    if blockSize%2 == 0:
        blockSize += 1 

    # Dissolve noise while maintaining edge sharpness 
    bgr2gray = cv2.bilateralFilter( bgr2gray, 5, 17, 17 ) #( bgr2gray, 11, 17, 17 )
    bgr2gray = cv2.GaussianBlur( bgr2gray, (5, 5), GaussianBlur ) #1

    # Threshold any color that is not black to white
    if threshType == 0:
        thresholded = cv2.adaptiveThreshold( bgr2gray, maxValue, cv2.ADAPTIVE_THRESH_MEAN_C,
                                             cv2.THRESH_BINARY, blockSize, cte )
    elif threshType == 1:
        thresholded = cv2.adaptiveThreshold( bgr2gray, maxValue, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             cv2.THRESH_BINARY, blockSize, cte )
    elif threshType == 2:
        thresholded = cv2.adaptiveThreshold( bgr2gray, maxValue, cv2.ADAPTIVE_THRESH_MEAN_C,
                                             cv2.THRESH_BINARY_INV, blockSize, cte )
    elif threshType == 3:
        thresholded = cv2.adaptiveThreshold( bgr2gray, maxValue, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             cv2.THRESH_BINARY_INV, blockSize, cte )

    kernel = cv2.getStructuringElement( cv2.MORPH_RECT, ( 10, 10 ) )
    bgr2gray = cv2.erode( cv2.dilate( thresholded, kernel, iterations=1 ), kernel, iterations=1 )

    # Place processed image in queue for retrieval
    Q_procFrame.put(bgr2gray)


# ******************************************************
# Define a function to get distance from ToF sensor
# ******************************************************
def getDist():

    # No need to reevaluate in the main function at every iteration
    global ToF_Dist

    # Listen to serial port as long as port is open
    while ( ToF.is_open ):

        # Do the reading iff there is something available at serial port
        if ToF.in_waiting > 0:
            ToF_Dist = int( (ToF.read(size=1).strip('\0')).strip('\n') )
            # If debug flag is invoked
            if args["debug"]:
                print( ToF_Dist )
        else:
            pass


# ******************************************************
# Define a function to scan for circles from camera feed
# ******************************************************
def scan4circles( bgr2gray, overlay, overlayImg, frame, Q_scan4circles ):

    # Error handling in case a non-allowable integer is chosen (1)
    try:
        # Scan for circles
        circles = cv2.HoughCircles( bgr2gray, cv2.HOUGH_GRADIENT, dp, minDist,
                                    param1, param2, minRadius, maxRadius )

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

            # If within scan distance display found circles
            if ToF_Dist == 1:
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
                    Q_scan4circles.put( output )

    # Error handling in case a non-allowable integer is chosen (2)
    except Exception as instance:
        print( fullStamp() + " Exception or Error Caught" )
        print( fullStamp() + " Error Type %s" %str(type(instance)) )
        print( fullStamp() + " Resetting Trackbars..." )

        # Reset trackbars
        cv2.createTrackbar( "dp"        , ver, 34   , 50 , placeholder ) #14
        cv2.createTrackbar( "minDist"   , ver, 396  , 750, placeholder )
        cv2.createTrackbar( "param1"    , ver, 316  , 750, placeholder ) #326
        cv2.createTrackbar( "param2"    , ver, 236  , 750, placeholder ) #231
        cv2.createTrackbar( "minRadius" , ver, 7    , 200, placeholder ) #1
        cv2.createTrackbar( "maxRadius" , ver, 14   , 250, placeholder )

        print( fullStamp() + " Success" )

        # Exit function and re-loop
        return()

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

cv2.createTrackbar( "Type:\n0.MEAN_Binary\n1.GAUSSIAN_Binary\n2.MEAN_BinaryInv\n3.GAUSSIAN_BinaryInv\n4.OTSU",
                    "AI_View", 2, 3, placeholder )
cv2.createTrackbar( "maxValue", "AI_View", 138, 255, placeholder )
cv2.createTrackbar( "blockSize", "AI_View", 180, 254, placeholder ) #45
cv2.createTrackbar( "cte", "AI_View", 59, 100, placeholder )
cv2.createTrackbar( "GaussianBlur", "AI_View", 0, 50, placeholder )

# Initialize ToF sensor
deviceName, port, baudRate = "VL6180", 0, 115200
ToF = createUSBPort( deviceName, port, baudRate, 3 )
if ToF.is_open == False:
    ToF.open()
    sleep( 0.5 )
    inChar = (ToF.read(size=1).strip('\0')).strip('\n')
    ToF.write('2')
    while inChar is not 'y':
        inChar = (ToF.read(size=1).strip('\0')).strip('\n')
        # If debug flag is invoked
        if args["debug"]:
            print( inChar )
    print( fullStamp() + " Distance Readings Initiated" )

ToF_Dist = 0    # Initialize to OFF

# Create a queue for retrieving data from thread
Q_B_procFrame = Queue( maxsize=0 )
Q_G_procFrame = Queue( maxsize=0 )
Q_R_procFrame = Queue( maxsize=0 )
Q_scan4circles  = Queue( maxsize=0 )

# Start listening to serial port
t_getDist = Thread( target=getDist, args=() )
t_getDist.daemon = True
t_getDist.start()

# If debug flag is invoked
if args["debug"]:
    # Start benchmark
    print( fullStamp() + " [INFO] Debug Mode: ON" )
    fps = FPS().start()

# Windows to visualize RGB channels
cv2.namedWindow( "B" )
cv2.namedWindow( "G" )
cv2.namedWindow( "R" )
# ************************************************************************
# =========================> MAKE IT ALL HAPPEN <=========================
# ************************************************************************

# Infinite loop
while True:
    
    # Get image from stream
    frame = stream.read()[36:252, 48:336]

    # Add a 4th dimension (Alpha) to the captured frame
    (h, w) = frame.shape[:2]
    frame = numpy.dstack( [frame, numpy.ones( ( h, w ), dtype="uint8" ) * 255] )
    output = frame

    # Create an overlay layer
    overlay = numpy.zeros( ( h, w, 4 ), "uint8" )

    # Split frame into multiple channels
    B, G, R, A = frame[:,:,0], frame[:,:,1], frame[:,:,2], frame[:,:,3]
    M = numpy.maximum( numpy.maximum( R, G ), B )
    R[ R < M]=0
    G[ G < M]=0
    B[ B < M]=0
    cv2.imshow("B", B)
    cv2.imshow("G", G)
    cv2.imshow("R", R)
    
    # Start thread to process image and apply required filters to detect circles
    # B Thread
    t_B_procFrame = Thread( target=procFrame, args=( B, Q_B_procFrame ) )
    t_B_procFrame.start()
    # G Thread
    t_G_procFrame = Thread( target=procFrame, args=( G, Q_G_procFrame ) )
    t_G_procFrame.start()
    # R Thread
    t_R_procFrame = Thread( target=procFrame, args=( R, Q_R_procFrame ) )
    t_R_procFrame.start()

    # Check if queue has something available for retrieval
    if Q_B_procFrame.qsize() > 0:
        B = Q_B_procFrame.get()

    if Q_G_procFrame.qsize() > 0:
        G = Q_G_procFrame.get()

    if Q_R_procFrame.qsize() > 0:
        R = Q_R_procFrame.get()

    
    # Combine images using bitwise AND to avoid over saturation
    B = cv2.bitwise_and( B, B, mask=A )
    G = cv2.bitwise_and( G, G, mask=A )
    R = cv2.bitwise_and( R, R, mask=A )
    frame = cv2.merge( [B, G, R, A] )
    
    # Convert into grayscale because HoughCircle only accepts grayscale images
    bgr2gray = cv2.cvtColor( frame, cv2.COLOR_BGR2GRAY )


    # Get trackbar position and reflect it in HoughCircles parameters input
    dp = cv2.getTrackbarPos( "dp", ver )
    minDist = cv2.getTrackbarPos( "minDist", ver )
    param1 = cv2.getTrackbarPos( "param1", ver )
    param2 = cv2.getTrackbarPos( "param2", ver )
    minRadius = cv2.getTrackbarPos( "minRadius", ver )
    maxRadius = cv2.getTrackbarPos( "maxRadius", ver )

    # Start thread to scan for circles
    t_scan4circles = Thread( target=scan4circles, args=( bgr2gray, overlay, overlayImg, output, Q_scan4circles ) )
    t_scan4circles.start()

    # Check if queue has something available for retrieval
    if Q_scan4circles.qsize() > 0:
        output = Q_scan4circles.get()

    # If debug flag is invoked
    if args["debug"]:
       fps.update()

    # Live feed display toggle
    if normalDisplay:
        cv2.imshow("B", B)
        cv2.imshow("G", G)
        cv2.imshow("R", R)
        cv2.imshow(ver, output)
        cv2.imshow( "AI_View", bgr2gray )
        key = cv2.waitKey(1) & 0xFF
    elif not(normalDisplay):
        cv2.imshow(ver, frame)
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

------------------------------------------------------------------------------------------------

    # Check if queue has something available for retrieval
    if Q_procFrame.qsize() > 0:
        bgr2gray = Q_procFrame.get()
##        bgr2grayBAK = bgr2gray
##        if initRun==False:
##            if abs(oldx - x_ROI) > 10 or abs(oldy - y_ROI) > 10:
##                # Calculate Region of interest constraints
##                x1_ROI = x_ROI-r_ROI-10
##                x2_ROI = x_ROI+r_ROI+10
##                y1_ROI = y_ROI-r_ROI-10
##                y2_ROI = y_ROI+r_ROI+10
##
##                oldx = x_ROI
##                oldy = y_ROI
##                print"updated xyROI"
##
##            if x1_ROI>0 and x1_ROI<w and x2_ROI>0 and x2_ROI<w and y1_ROI>0 and y1_ROI<h and y2_ROI>0 and y2_ROI<h:
##                print("xyr_ROI: " , (x_ROI, y_ROI, r_ROI))
##                print("ROI Location: ", (y_ROI-r_ROI), (y_ROI+r_ROI), (x_ROI-r_ROI), (x_ROI+r_ROI))
##                print("BGR shape: ", bgr2gray.shape)
##                bgr2gray = bgr2gray[ y1_ROI:y2_ROI, x1_ROI:x2_ROI]
##            else:
##                bgr2gray = bgr2grayBAK

    # Get trackbar position and reflect it in HoughCircles parameters input
    dp = cv2.getTrackbarPos( "dp", ver )
    minDist = cv2.getTrackbarPos( "minDist", ver )
    param1 = cv2.getTrackbarPos( "param1", ver )
    param2 = cv2.getTrackbarPos( "param2", ver )
    minRadius = cv2.getTrackbarPos( "minRadius", ver )
    maxRadius = cv2.getTrackbarPos( "maxRadius", ver )

-----------------------------------------------------------------------------------------------------

    # Check if queue has something available for retrieval
    if Q_scan4circles.qsize() > 0:
        output, (x_ROI, y_ROI, r_ROI) = Q_scan4circles.get()
##        if initRun==True:
##            initRun=False
##
##            x1_ROI = x_ROI-r_ROI-10
##            x2_ROI = x_ROI+r_ROI+10
##            y1_ROI = y_ROI-r_ROI-10
##            y2_ROI = y_ROI+r_ROI+10
##
##            oldx = x_ROI
##            oldy = y_ROI

    # If debug flag is invoked
    if args["debug"]:
       fps.update()

'''
