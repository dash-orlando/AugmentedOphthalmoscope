'''
* NOTE: Version 1.0b is rewritten from scratch to improve on the
*       shortcomings of the previous version (mainly too many false
*       positives & general instability).
*       Main change was in the identification method, which now uses
*       findContours() instead of HoughCircles().
*
* USEFUL ARGUMENTS:
*   -o/--overlay: Specify overlay file
*   -a/--alpha  : Specify transperancy level (0.0 - 1.0)
*   -d/--debug  : Enable debugging
*
* VERSION: 1.0b
*   - ADDED   : Initial release 
*
* KNOWN ISSUES:
*   - Code is VERY buggy and lacks many of the previous
*     functionalities (i.e overlays, threading, etc...)
*
* AUTHOR                    :   Mohammad Odeh
* WRITTEN                   :   Aug   1st, 2016 Year of Our Lord
* LAST CONTRIBUTION DATE    :   May. 21st, 2018 Year of Our Lord
*
* ----------------------------------------------------------
* ----------------------------------------------------------
*
* RIGHT CLICK: Shutdown Program.
* LEFT CLICK: Toggle view.
'''

ver = "Live Feed Ver1.0b"
print __doc__

import  cv2, argparse                                                   # Various Stuff
import  numpy                                               as  np      # Image manipulation
from    timeStamp                   import  fullStamp       as  FS      # Show date/time on console output
from    imutils.video.pivideostream import  PiVideoStream               # Import threaded PiCam module
from    imutils.video               import  FPS                         # Benchmark FPS
from    time                        import  sleep                       # Sleep for stability
from    threading                   import  Thread                      # Used to thread processes
from    Queue                       import  Queue                       # Used to queue input/output
from    usbProtocol                 import  createUSBPort               # Create USB Port
from    LEDRing                     import  *                           # Let there be light

# ************************************************************************
# =====================> CONSTRUCT ARGUMENT PARSER <=====================*
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
def control( event, x, y, flags, param ):
    '''
    Left/right click mouse events
    '''
    global normalDisplay
    
    # Right button shuts down program
    if( event == cv2.EVENT_RBUTTONDOWN ):

        # Do some shutdown clean up
        try:
            colorWipe(strip, Color(0, 0, 0, 0), 0)                      # Turn OFF LED ring
            stream.stop()                                               # Stop capturing frames from stream
            cv2.destroyAllWindows()                                     # Close any open windows
            
        except Exception as instance:
            print( "{} Error caught in control()".format(FS()) )
            print( "{0} {1}".format(FS(), type(instance)) )
            pass

        finally: 
            quit()                                                      # Shutdown python interpreter
     
    # Left button toggles display
    elif( event == cv2.EVENT_LBUTTONDOWN ):
        normalDisplay = not( normalDisplay )


# ------------------------------------------------------------------------

def placeholder( x ):
    '''
    Place holder function for trackbars. This is required
    for the trackbars to function properly.
    '''
    
    pass

# ------------------------------------------------------------------------

def procFrame( image ):
    '''
    Process frame by applying a bilateral filter, a Gaussian blur,
    and an adaptive threshold + some post-processing

    INPUTS:-
        - image: Grayscale image to be processed

    OUTPUT:-
        - bgr2gray: Processed image
    '''

    try:
        bgr2gray = cv2.inRange(image, lower_bound, upper_bound)
        # Get trackbar position and reflect its threshold type and values
        threshType = cv2.getTrackbarPos( "Type:\n0.MEAN_Binary\n1.GAUSSIAN_Binary\n2.MEAN_BinaryInv\n3.GAUSSIAN_BinaryInv\n4.OTSU",
                                         "AI_View")
        maxValue        = cv2.getTrackbarPos( "maxValue"    , "AI_View")
        blockSize       = cv2.getTrackbarPos( "blockSize"   , "AI_View")
        cte             = cv2.getTrackbarPos( "cte"         , "AI_View")
        GaussianBlur    = cv2.getTrackbarPos( "GaussianBlur", "AI_View")

        # blockSize must be an odd number
        if blockSize%2 == 0:
            blockSize += 1 

        # Dissolve noise while maintaining edge sharpness 
        bgr2gray = cv2.bilateralFilter( bgr2gray, 5, 17, 17 ) #( bgr2gray, 11, 17, 17 )
        bgr2gray = cv2.GaussianBlur( bgr2gray, (5, 5), GaussianBlur ) #1

        # Threshold any color that is not black to white (or vice versa)
        if( threshType == 0 ):
            thresholded = cv2.adaptiveThreshold( bgr2gray, maxValue,
                                                 cv2.ADAPTIVE_THRESH_MEAN_C,
                                                 cv2.THRESH_BINARY, blockSize, cte )
        elif( threshType == 1 ):
            thresholded = cv2.adaptiveThreshold( bgr2gray, maxValue,
                                                 cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                 cv2.THRESH_BINARY, blockSize, cte )
        elif( threshType == 2 ):
            thresholded = cv2.adaptiveThreshold( bgr2gray, maxValue,
                                                 cv2.ADAPTIVE_THRESH_MEAN_C,
                                                 cv2.THRESH_BINARY_INV, blockSize, cte )
        elif( threshType == 3 ):
            thresholded = cv2.adaptiveThreshold( bgr2gray, maxValue,
                                                 cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                 cv2.THRESH_BINARY_INV, blockSize, cte )

        kernel = np.ones((3, 3), np.uint8)

        # Use erosion and dilation combination to eliminate false positives. 
        bgr2gray = cv2.erode ( thresholded, kernel, iterations=6 )          # Erode over 6 passes
        bgr2gray = cv2.dilate( bgr2gray, kernel, iterations=3 )             # Dilate over 3 passes

        morphed  = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)      # Morph image for shits and gigs
        
        return( bgr2gray, morphed )                                         # Return
    
    except:
        cv2.createTrackbar( "maxValue"      , "AI_View", 245, 255, placeholder ) #138
        cv2.createTrackbar( "blockSize"     , "AI_View", 254, 254, placeholder ) #180
        cv2.createTrackbar( "cte"           , "AI_View", 65 , 100, placeholder ) #59
        cv2.createTrackbar( "GaussianBlur"  , "AI_View", 12 , 50 , placeholder ) #0
# ------------------------------------------------------------------------

def scan4circles( bgr2gray, overlay, overlayImg, frame ):
    '''
    Scan for circles within an image

    INPUTS:
        - bgr2gray  : Processed grayscale image
        - overlay   : The overlay empty frame
        - overlayImg: The overlay image
        - frame     : Frame to which we should attach overlay

    OUTPUT:
        - output    : Image with/without overlay
                (depends whether circles were found or not)
    '''
    
    # Error handling in case a non-allowable integer is chosen (1)
    try:
        r_min   = cv2.getTrackbarPos( "minRadius", ver )                    # Get current r_min ...
        r_max   = cv2.getTrackbarPos( "maxRadius", ver )                    # and r_max values
         
        im2, contours, hierarchy = cv2.findContours( bgr2gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE )

        # If circles are found draw them
        if contours is not None:
            for c in contours:                                              # Iterate over all contours
                (x,y),r = cv2.minEnclosingCircle(c)                         # 
                center = (int(x),int(y))                                    # Get circle's center...
                r = int(r)                                                  # and radius
                
                if( r_min <= r and r <= r_max ):                            # Check if within desired limit
                    output = cv2.addWeighted( overlay,                      # Join overlay with the...
                                              args["alpha"],                # livefeed frame and apply...
                                              frame, 1.0, 0 )               # specified alpha levels.
                    cv2.circle( frame, center, r, (0, 255, 0), 2 )          # draw a circle

                else:
                    output = frame
        else:
            output = frame

        return( output )                            

    # Error handling in case a non-allowable integer is chosen (2)
    except Exception as instance:
##        print( "{} Error caught in scan4circles()".format(FS()) )
##        print( "{0} {1}".format(FS(), instance) )

        # Reset trackbars
        cv2.createTrackbar( "minRadius" , ver, 15   , 150, placeholder ) #7
        cv2.createTrackbar( "maxRadius" , ver, 25   , 150, placeholder ) #14

        # Exit function and re-loop
        return( None )
    

# ************************************************************************
# ===========================> SETUP PROGRAM <===========================
# ************************************************************************

# Check whether an overlay is specified
if( args["overlay"] != None ):                                              # If overlay is defined ...
    overlayImg = cv2.imread( args["overlay"], cv2.IMREAD_UNCHANGED )        # Load specific overlay
else:                                                                       # Else ...
    overlayImg = cv2.imread( "Overlay.png"  , cv2.IMREAD_UNCHANGED )        # Load default overlay

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
normalDisplay = True                                                        # Start with a normal display
colorWipe(strip, Color(255, 255, 255, 255), 0)                              # Turn ON LED ring

# Setup main window
cv2.namedWindow( ver )                                                      # Start a named window for output
cv2.setMouseCallback( ver, control )                                        # Connect mouse events to actions
cv2.createTrackbar( "minRadius" , ver, 15   , 150, placeholder )            # Trackbars for min and max radii
cv2.createTrackbar( "maxRadius" , ver, 25   , 150, placeholder )            #     (used in scan4circles)

# Setup window and trackbars for AI view
cv2.namedWindow( "AI_View" )

cv2.createTrackbar( "Type:\n0.MEAN_Binary\n1.GAUSSIAN_Binary\n2.MEAN_BinaryInv\n3.GAUSSIAN_BinaryInv\n4.OTSU",
                    "AI_View", 3, 3, placeholder )
cv2.createTrackbar( "maxValue"      , "AI_View", 245, 255, placeholder )
cv2.createTrackbar( "blockSize"     , "AI_View", 230, 254, placeholder )
cv2.createTrackbar( "cte"           , "AI_View", 8  , 100, placeholder )
cv2.createTrackbar( "GaussianBlur"  , "AI_View", 26 , 50 , placeholder )
    
# ************************************************************************
# =========================> MAKE IT ALL HAPPEN <=========================
# ************************************************************************
global lower_bound, upper_bound

lower_bound = np.array([0,0,10])
upper_bound = np.array([255,255,195])

while( True ):
    frame= stream.read()[36:252, 48:336]
    image_ori = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    image = frame
    
    # Add a 4th dimension (Alpha) to the captured frame
    (h, w) = frame.shape[:2]                                        # Determine width and height
    frame = np.dstack([ frame, np.ones((h, w),                      # Stack the arrays in sequence, 
                                       dtype="uint8")*255 ])        # depth-wise ( along the z-axis )

    # Create an overlay layer
    overlay = np.zeros( ( h, w, 4 ), "uint8" )                      # Empty np array w\ same dimensions as the frame

    mask, closing = procFrame( image )
    
    # Check if queue has something available for retrieval
    out = scan4circles( mask, overlay, overlayImg, frame )
    if( out is not None ):
        image = out
    else:
        pass
        
    # Live feed display toggle
    if normalDisplay:
        cv2.imshow( ver, image)
        cv2.imshow( "AI_View", mask )
        key = cv2.waitKey(1) & 0xFF
    elif not(normalDisplay):
        cv2.imshow(ver, closing)
        cv2.imshow( "AI_View", mask )
        key = cv2.waitKey(1) & 0xFF
        
