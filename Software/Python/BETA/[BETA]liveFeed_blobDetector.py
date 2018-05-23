'''
* NOTE: Version 1.0a is rewritten from scratch to improve on the
*       shortcomings of the previous version (mainly too many false
*       positives & general instability).
*       Main change was in the identification method, which now uses
*       SimpleBlobDetector_create() ==> detector.detect(img) instead
*       of HoughCircles().
*
* USEFUL ARGUMENTS:
*   -o/--overlay: Specify overlay file
*   -a/--alpha  : Specify transperancy level (0.0 - 1.0)
*   -d/--debug  : Enable debugging
*
* VERSION: 1.0.1a
*   - ADDED   : Overlay an image/pathology
*   - MODIFIED: Significantly reduced false-positives
*   - ADDED   : Define a ROI. If blobs are detected outside
*               ROI,ignore them
*
* KNOWN ISSUES:
*   - Code is VERY buggy and lacks many of the previous
*     functionalities (i.e threading, etc...)
*   - Unable to capture pupil when looking from the side
*
* AUTHOR                    :   Mohammad Odeh
* WRITTEN                   :   Aug   1st, 2016 Year of Our Lord
* LAST CONTRIBUTION DATE    :   May. 23rd, 2018 Year of Our Lord
*
* ----------------------------------------------------------
* ----------------------------------------------------------
*
* RIGHT CLICK: Shutdown Program.
* LEFT CLICK: Toggle view.
'''

ver = "Live Feed Ver1.0.1a"
print __doc__

import  cv2, argparse                                                       # Various Stuff
import  numpy                                               as  np          # Image manipulation
from    timeStamp                   import  fullStamp       as  FS          # Show date/time on console output
from    imutils.video.pivideostream import  PiVideoStream                   # Import threaded PiCam module
from    imutils.video               import  FPS                             # Benchmark FPS
from    LEDRing                     import  *                               # Let there be light
from    time                        import  sleep, time                     # Time is essential to life

# ************************************************************************
# =====================> CONSTRUCT ARGUMENT PARSER <=====================*
# ************************************************************************
ap = argparse.ArgumentParser()

ap.add_argument( "-o", "--overlay", required=False,
                 help="Path to overlay image" )

ap.add_argument( "-a", "--alpha", type=float, default=0.85,
                 help="Set alpha level.\nDefault=0.85" )

ap.add_argument( "-d", "--debug", action='store_true',
                 help="Enable debugging" )

args = vars( ap.parse_args() )

args["debug"] = True
# ************************************************************************
# =====================> DEFINE NECESSARY FUNCTIONS <=====================
# ************************************************************************
def control( event, x, y, flags, param ):
    '''
    Left/right click mouse events
    '''
    global realDisplay
    
    # Right button shuts down program
    if( event == cv2.EVENT_RBUTTONDOWN ):                                   # If right-click, do shutdown clean up
        try:
            colorWipe(strip, Color(0, 0, 0, 0), 0)                          # Turn OFF LED ring
            stream.stop()                                                   # Stop capturing frames from stream
            cv2.destroyAllWindows()                                         # Close any open windows
            
        except Exception as error:
            if( ["debug"] ):
                print( "{} Error caught in control()".format(FS()) )        # Specify error type
                print( "{0} {1}".format(FS(), type(error)) )                # ...

        finally: 
            quit()                                                          # Shutdown python interpreter
     
    # Left button toggles display
    elif( event == cv2.EVENT_LBUTTONDOWN ):                                 # If left click, switch display
        realDisplay = not( realDisplay )                                    # ...

# ------------------------------------------------------------------------

def placeholder( x ):
    '''
    Place holder function for trackbars. This is required
    for the trackbars to function properly.
    '''
    
    pass

# ------------------------------------------------------------------------

def update_blobDetector( x ):
    '''
    Update blob detector parameters. Gets called whenever
    trackbars are updated

    INPUTS:-
        - x: nothing

    OUTPUT:-
        - Non
    '''
    
    global detector, params

    r_min       = cv2.getTrackbarPos( "minRadius"    , ver )                # Get updated blob detector
    r_max       = cv2.getTrackbarPos( "maxRadius"    , ver )                # parameters
    circle_min  = cv2.getTrackbarPos( "Circularity"  , ver )                # ...
    convex_min  = cv2.getTrackbarPos( "Convexity"    , ver )                # ...
    inertia_min = cv2.getTrackbarPos( "InertiaRatio" , ver )                # ...

    params.minArea = np.pi * r_min**2                                       # Update parameters
    params.maxArea = np.pi * r_max**2                                       # ...
    params.minCircularity   = circle_min/100.                               # ...
    params.minConvexity     = convex_min/100.                               # ...
    params.minInertiaRatio  = inertia_min/100.                              # ...
    
    detector = cv2.SimpleBlobDetector_create( params )                      # Reflect on detector
    
# ------------------------------------------------------------------------

def procFrame( image ):
    '''
    Process frame by applying a bilateral filter, a Gaussian blur,
    and an adaptive threshold + some post-processing

    INPUTS:-
        - image     : Image to be processed

    OUTPUT:-
        - processed : Processed image
    '''

    try:
        # Get trackbar position and reflect its threshold type and values
        threshType = cv2.getTrackbarPos( "Type:\n0.MEAN_Binary\n1.GAUSSIAN_Binary\n2.MEAN_BinaryInv\n3.GAUSSIAN_BinaryInv",
                                         "CV")                              # ...
        maxValue        = cv2.getTrackbarPos( "maxValue"    , "CV")         # Update parameters
        blockSize       = cv2.getTrackbarPos( "blockSize"   , "CV")         # ...
        cte             = cv2.getTrackbarPos( "cte"         , "CV")         # from trackbars
        GaussianBlur    = cv2.getTrackbarPos( "GaussianBlur", "CV")         # ...

        # blockSize must be an odd number
        if( blockSize%2 == 0 ):                                             # Ensure that blockSize
            blockSize += 1                                                  # is an odd number
        
        # Dissolve noise while maintaining edge sharpness
        processed = cv2.inRange( image, lower_bound, upper_bound )
        processed = cv2.bilateralFilter( processed, 5, 17, 17 ) #( processed, 11, 17, 17 )
        processed = cv2.GaussianBlur( processed, (5, 5), GaussianBlur ) #1

        # Threshold any color that is not black to white (or vice versa)
        if( threshType == 0 ):
            processed = cv2.adaptiveThreshold( processed, maxValue,
                                               cv2.ADAPTIVE_THRESH_MEAN_C,
                                               cv2.THRESH_BINARY, blockSize, cte )
        elif( threshType == 1 ):
            processed = cv2.adaptiveThreshold( processed, maxValue,
                                               cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                               cv2.THRESH_BINARY, blockSize, cte )
        elif( threshType == 2 ):
            processed = cv2.adaptiveThreshold( processed, maxValue,
                                               cv2.ADAPTIVE_THRESH_MEAN_C,
                                               cv2.THRESH_BINARY_INV, blockSize, cte )
        elif( threshType == 3 ):
            processed = cv2.adaptiveThreshold( processed, maxValue,
                                               cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                               cv2.THRESH_BINARY_INV, blockSize, cte )

        kernel = np.ones((3, 3), np.uint8)

        # Use erosion and dilation combination to eliminate false positives. 
        processed = cv2.erode ( processed, kernel, iterations=6 )           # Erode over 6 passes
        processed = cv2.dilate( processed, kernel, iterations=3 )           # Dilate over 3 passes

        morphed  = cv2.morphologyEx( image, cv2.MORPH_GRADIENT, kernel )    # Morph image for shits and gigs
        
        return( processed, morphed )                                        # Return
    
    except:
        cv2.createTrackbar( "maxValue"      , "CV", 245, 255, placeholder ) # Reset trackbars
        cv2.createTrackbar( "blockSize"     , "CV", 58, 254, placeholder ) # ...
        cv2.createTrackbar( "cte"           , "CV", 0 , 100, placeholder ) # ...
        cv2.createTrackbar( "GaussianBlur"  , "CV", 40 , 50 , placeholder ) # ...
        
# ------------------------------------------------------------------------

def scan4circles( processed, overlay_frame, overlay_img, frame ):
    '''
    Scan for circles within an image

    INPUTS:-
        - processed     : Processed image
        - overlay_frame : The overlay empty frame
        - overlay_img   : The overlay image
        - frame         : Frame to which we should attach overlay

    OUTPUT:-
        - frame         : Image with/without overlay
                (depends whether circles were found or not)
    '''
    global ROI, startTime
    
    ROI_x_min, ROI_y_min = ROI[0]
    ROI_x_max, ROI_y_max = ROI[1]
    # Error handling in case a non-allowable integer is chosen (1)
    try:

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)                      # Convert to grayscale
        keypoints = detector.detect(gray)                                   # Launch blob detector

        if( len(keypoints) > 0 ):                                           # If blobs are found
            for k in keypoints:                                             # Iterate over found blobs
                x, y, r = int(k.pt[0]), int(k.pt[1]), int(k.size/2)         # Get co-ordinates
                pos = ( x, y, r )                                           # Pack co-ordinates

                if( ROI_x_min < (x-r) and (x+r) < ROI_x_max ):              # Check if we are within ROI
                    if( ROI_y_min < (y-r) and (y+r) < ROI_y_max ):          # ...
                    
                        ROI[0] = ( (x-r-dx), (y-r-dy) )
                        ROI[1] = ( (x+r+dx), (y+r+dy) )
                        startTime = time()
                        frame = add_overlay( overlay_frame,                 # Add overlay
                                             overlay_img,                   # ...
                                             frame, pos )                   # ...
                        
                    else:
                        if( time() - startTime >= timeout ):
                            print( "ROI Reset" )
                            startTime = time()
                            ROI = ROI_0[:]                                  # Reset ROI if needed be
                else:
                    if( time() - startTime >= timeout ):
                        print( "ROI Reset" )
                        startTime = time()
                        ROI = ROI_0[:]                                      # Reset ROI if needed be
        else:
            if( time() - startTime >= timeout ):
                print( "ROI Reset" )
                startTime = time()
                ROI = ROI_0[:]                                              # Reset ROI if needed be

        return( frame )

    # Error handling in case a non-allowable integer is chosen (2)
    except Exception as error:
        if( ["debug"] ):
            print( "{} Error caught in scan4circles()".format(FS()) )       # Specify error type
            print( "{0} {1}".format(FS(), error) )                          # ...

        cv2.createTrackbar( "minRadius"    , ver, 10 , 150, update_blobDetector )   # Reset trackbars
        cv2.createTrackbar( "maxRadius"    , ver, 40 , 150, update_blobDetector )   # ...
        cv2.createTrackbar( "Circularity"  , ver, 25 , 100, update_blobDetector )   # ...
        cv2.createTrackbar( "Convexity"    , ver, 15 , 100, update_blobDetector )   # ...
        cv2.createTrackbar( "InertiaRatio" , ver, 70 , 100, update_blobDetector )   # ...

        return( None )                                                      # Return NONE
    
# ------------------------------------------------------------------------

def add_overlay( overlay_frame, overlay_img, frame, pos ):
    '''
    Resize and add overlay image into detected pupil location 

    INPUTS:
        - overlay_frame : The overlay empty frame
        - overlay_img   : The overlay image
        - frame         : Frame to which we should attach overlay
        - pos           : Co-ordinates where pupil is

    OUTPUT:
        - frame         : Image with overlay
    '''
    
    x, y, r = pos                                                           # Unpack co-ordinates
    
    x_min, x_max = x-r, x+r                                                 # Find min/max x-range
    y_min, y_max = y-r, y+r                                                 # Find min/max y-range

    if( x_min > 0 and y_min > 0 ):
        if( x_max < w and y_max < h ):
            overlay_img = cv2.resize( overlay_img, ( 2*r, 2*r ),            # Resize overlay image to fit
                                      interpolation = cv2.INTER_AREA )      # ...
            
            overlay_frame[ y_min:y_max, x_min:x_max] = overlay_img          # Place overlay image into overlay frame

            frame = cv2.addWeighted( overlay_frame,                         # Join overlay frame (alpha)
                                     args["alpha"],                         # with actual frame (RGB)
                                     frame, 1.0, 0 )                        # ...

            if( ["debug"] ):
                cv2.circle( frame, (x, y), r, (0, 255, 0), 2 )              # Draw a circle
                cv2.rectangle( frame, ROI[0], ROI[1], (255, 0, 0), 2 )
            
    return( frame )
        
# ------------------------------------------------------------------------

def reset_ROI():
    '''
    Dynamically tracka/update ROI where pupil is located

    INPUTS:
        - blah1 : ..

    OUTPUT:
        - blah2 : ...
    '''

# ************************************************************************
# ===========================> SETUP PROGRAM <===========================
# ************************************************************************

######
### General Setup
######
# Check whether an overlay is specified
if( args["overlay"] != None ):
    overlayImg = cv2.imread( args["overlay"], cv2.IMREAD_UNCHANGED )        # Load specific overlay
else:
    src = "/home/pi/Desktop/BETA/Overlay.png"                               # Load default overlay
    overlayImg = cv2.imread( src  , cv2.IMREAD_UNCHANGED )                  # ...

# Load overlay image with Alpha channel
( wH, wW ) = overlayImg.shape[:2]                                           # Get dimensions
( B, G, R, A ) = cv2.split( overlayImg )                                    # Split into constituent channels
B = cv2.bitwise_and( B, B, mask=A )                                         # Add the Alpha to the B channel
G = cv2.bitwise_and( G, G, mask=A )                                         # Add the Alpha to the G channel
R = cv2.bitwise_and( R, R, mask=A )                                         # Add the Alpha to the R channel
overlayImg = cv2.merge( [B, G, R, A] )                                      # Finally, merge them back

# Setup camera (x,y)
stream = PiVideoStream( resolution=(384, 288) ).start()                     # Start PiCam
sleep( 0.25 )                                                               # Sleep for stability
realDisplay = True                                                          # Start with a normal display
colorWipe(strip, Color(255, 255, 255, 255), 0)                              # Turn ON LED ring

# Setup main window
cv2.namedWindow( ver )                                                      # Start a named window for output
cv2.setMouseCallback( ver, control )                                        # Connect mouse events to actions
cv2.createTrackbar( "minRadius"     , ver, 10 , 100, update_blobDetector )  # Trackbars for blob detector
cv2.createTrackbar( "maxRadius"     , ver, 40 , 100, update_blobDetector )  # parameters.
cv2.createTrackbar( "Circularity"   , ver, 25 , 100, update_blobDetector )#50  #     (used in scan4circles)
cv2.createTrackbar( "Convexity"     , ver, 15 , 100, update_blobDetector )#25  # ...
cv2.createTrackbar( "InertiaRatio"  , ver, 70 , 100, update_blobDetector )#50  # ...

# Setup window and trackbars for CV window
cv2.namedWindow( "CV" )                                                     # Start a named window for CV

cv2.createTrackbar( "Type:\n0.MEAN_Binary\n1.GAUSSIAN_Binary\n2.MEAN_BinaryInv\n3.GAUSSIAN_BinaryInv",
                    "CV", 1, 3, placeholder )#3                               # ...
cv2.createTrackbar( "maxValue"      , "CV", 255, 255, placeholder )         # Trackbars to modify adaptive
cv2.createTrackbar( "blockSize"     , "CV", 58 , 254, placeholder )#130         # thresholding parameters
cv2.createTrackbar( "cte"           , "CV",  0 , 100, placeholder )#11         #   (used in procFrame)
cv2.createTrackbar( "GaussianBlur"  , "CV", 40 , 50 , placeholder )#16         # ...

######
### Setup BlobDetector
######
global detector, params

params = cv2.SimpleBlobDetector_Params()
	 
# Filter by Area.
params.filterByArea = True
params.minArea = 3.141 * 15**2
params.maxArea = 3.141 * 50**2

# Filter by color
params.filterByColor = True
params.blobColor = 0

# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.1
 
# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.25
	 
# Filter by Inertia
params.filterByInertia = True
params.minInertiaRatio = 0.8

# Distance Between Blobs
params.minDistBetweenBlobs = 1000
	 
# Create a detector with the parameters
detector = cv2.SimpleBlobDetector_create(params)

######
### Setup ROI
######
global startTime, ROI, dx, dy
startTime = time()
timeout = 2.5
dx, dy = 15, 15
ROI_0 = [ (144-50, 108-50), (144+50, 108+50) ]
ROI   = [ (144-50, 108-50), (144+50, 108+50) ]

# ************************************************************************
# =========================> MAKE IT ALL HAPPEN <=========================
# ************************************************************************
global lower_bound, upper_bound

##lower_bound = np.array( [0,0,10] )
##upper_bound = np.array( [255,255,195] )

lower_bound = np.array( [ 0, 0, 0] )
upper_bound = np.array( [75,75,75] )

while( True ):
    # Capture frame
    frame = stream.read()[36:252, 48:336]                                   # Capture frame and crop it
    image = frame                                                           # Save a copy of captured frame

    # Add a 4th dimension (Alpha) to the captured frame
    (h, w) = frame.shape[:2]                                                # Determine width and height
    frame = np.dstack([ frame, np.ones((h, w),                              # Stack the arrays in sequence, 
                                       dtype="uint8")*255 ])                # depth-wise ( along the z-axis )

    # Create an overlay layer
    overlay = np.zeros( ( h, w, 4 ), "uint8" )                              # Empty np array w\ same dimensions as the frame

    # Find circles
    mask, closing = procFrame( image )                                      # Process image
    out = scan4circles( mask, overlay, overlayImg, frame )                  # Scan pupil
    
    if( out is not None ):                                                  # If an output exists, update iamge
        image = out                                                         # ...
    else:                                                                   # Else, don't
        pass                                                                # ...

    if( ["debug"] ):
        cv2.rectangle( image, ROI_0[0], ROI_0[1], (0,0,255) ,2 )
        
    # Display feed
    if( realDisplay ):                                                      # Show real output
        cv2.imshow( ver, image)                                             # 
        cv2.imshow( "CV", mask )                                            # ...
        key = cv2.waitKey(1) & 0xFF                                         # ...
    else:                                                                   # Show morphed image
        cv2.imshow(ver, closing)                                            # ...
        cv2.imshow( "CV", mask )                                            # ...
        key = cv2.waitKey(1) & 0xFF                                         # ...
        
