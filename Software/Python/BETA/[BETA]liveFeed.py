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
* VERSION: 1.1.1b
*   - ADDED   : Overlay an image/pathology
*   - MODIFIED: Significantly reduced false-positives
*
* KNOWN ISSUES:
*   - Code is VERY buggy and lacks many of the previous
*     functionalities (i.e threading, etc...)
*
* AUTHOR                    :   Mohammad Odeh
* WRITTEN                   :   Aug   1st, 2016 Year of Our Lord
* LAST CONTRIBUTION DATE    :   May. 22nd, 2018 Year of Our Lord
*
* ----------------------------------------------------------
* ----------------------------------------------------------
*
* RIGHT CLICK: Shutdown Program.
* LEFT CLICK: Toggle view.
'''

ver = "Live Feed Ver1.0b"
print __doc__

import  cv2, argparse                                                       # Various Stuff
import  numpy                                               as  np          # Image manipulation
from    timeStamp                   import  fullStamp       as  FS          # Show date/time on console output
from    imutils.video.pivideostream import  PiVideoStream                   # Import threaded PiCam module
from    imutils.video               import  FPS                             # Benchmark FPS
from    time                        import  sleep                           # Sleep for stability
from    threading                   import  Thread                          # Used to thread processes
from    Queue                       import  Queue                           # Used to queue input/output
from    usbProtocol                 import  createUSBPort                   # Create USB Port
from    LEDRing                     import  *                               # Let there be light

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
        threshType = cv2.getTrackbarPos( "Type:\n0.MEAN_Binary\n1.GAUSSIAN_Binary\n2.MEAN_BinaryInv\n3.GAUSSIAN_BinaryInv\n4.OTSU",
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

        morphed  = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)      # Morph image for shits and gigs
        
        return( processed, morphed )                                        # Return
    
    except:
        cv2.createTrackbar( "maxValue"      , "CV", 245, 255, placeholder ) # Reset trackbars
        cv2.createTrackbar( "blockSize"     , "CV", 254, 254, placeholder ) # ...
        cv2.createTrackbar( "cte"           , "CV", 65 , 100, placeholder ) # ...
        cv2.createTrackbar( "GaussianBlur"  , "CV", 12 , 50 , placeholder ) # ...
        
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
    
    # Error handling in case a non-allowable integer is chosen (1)
    try:
        r_min   = cv2.getTrackbarPos( "minRadius", ver )                    # Get current r_min ...
        r_max   = cv2.getTrackbarPos( "maxRadius", ver )                    # and r_max values
         
        _, contours, _ = cv2.findContours( processed,                       # Find contours based on their...
                                           cv2.RETR_EXTERNAL,               # external edges and keep only...
                                           cv2.CHAIN_APPROX_SIMPLE )        # intermediate points (SIMPLE)
        
        # If circles are found draw them
        if( len(contours) > 0 ):                                            # Check if we detected anything
            for c in contours:                                              # Iterate over all contours
                (x,y),r = cv2.minEnclosingCircle(c)                         # 
                center = (int(x),int(y))                                    # Get circle's center...
                r = int(r)                                                  # and radius
                
                if( r_min <= r and r <= r_max ):                            # Check if within desired limit
                    pos = ( center, r )                                     # Pack co-ordinates
                    hull = cv2.convexHull(c)
##                    print( cv2.isContourConvex(c) )
##                    cv2.polylines( frame, hull, True, (0,255,0), 3)
                    frame = add_overlay( overlay_frame,                     # Add overlay
                                         overlay_img,                       # ...
                                         frame, pos, )                      # ...
                    
                else: pass                                                  # Nothing is found
        else: pass                                                          # ...

        return( frame )

    # Error handling in case a non-allowable integer is chosen (2)
    except Exception as error:
        if( ["debug"] ):
            print( "{} Error caught in scan4circles()".format(FS()) )       # Specify error type
            print( "{0} {1}".format(FS(), error) )                          # ...

        cv2.createTrackbar( "minRadius" , ver, 15   , 150, placeholder )    # Reset trackbars
        cv2.createTrackbar( "maxRadius" , ver, 35   , 150, placeholder )    # ...

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
    
    (x, y), r = pos                                                         # Unpack co-ordinates
    
    x_min, x_max = x-r, x+r                                                 # Find min/max x-range
    y_min, y_max = y-r, y+r                                                 # Find min/max y-range

    if( x_min > 0 and y_min > 0 ):
        if( x_max < w and y_max < h ):
##            gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
##            keypoints = detector.detect(gray)
##            for k in keypoints:
##            cv2.circle(frame, (int(k.pt[0]), int(k.pt[1])), int(k.size/2), (0, 255, 0), 2)
            overlay_img = cv2.resize( overlay_img, ( 2*r, 2*r ),            # Resize overlay image to fit
                                      interpolation = cv2.INTER_AREA )      # ...
            
            overlay_frame[ y_min:y_max, x_min:x_max] = overlay_img          # Place overlay image into overlay frame

            frame = cv2.addWeighted( overlay_frame,                         # Join overlay frame (alpha)
                                     args["alpha"],                         # with actual frame (RGB)
                                     frame, 1.0, 0 )                        # ...

            if( ["debug"] ):
                cv2.circle( frame, (x, y), r, (0, 255, 0), 2 )              # Draw a circle
            
    return( frame )

# ************************************************************************
# ===========================> SETUP PROGRAM <===========================
# ************************************************************************

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
cv2.createTrackbar( "minRadius" , ver, 15   , 150, placeholder )            # Trackbars for min and max radii
cv2.createTrackbar( "maxRadius" , ver, 35   , 150, placeholder )            #     (used in scan4circles)

# Setup window and trackbars for CV window
cv2.namedWindow( "CV" )                                                     # Start a named window for CV

cv2.createTrackbar( "Type:\n0.MEAN_Binary\n1.GAUSSIAN_Binary\n2.MEAN_BinaryInv\n3.GAUSSIAN_BinaryInv\n4.OTSU",
                    "CV", 0, 3, placeholder )#3                               # ...
cv2.createTrackbar( "maxValue"      , "CV", 255, 255, placeholder )         # Trackbars to modify adaptive
cv2.createTrackbar( "blockSize"     , "CV", 245, 254, placeholder )#230         # thresholding parameters
cv2.createTrackbar( "cte"           , "CV", 2  , 100, placeholder )#8         #   (used in procFrame)
cv2.createTrackbar( "GaussianBlur"  , "CV", 16 , 50 , placeholder )#26         # ...

### Setup BlobDetector
##detector = cv2.SimpleBlobDetector_create()
##params = cv2.SimpleBlobDetector_Params()
##	 
### Filter by Area.
##params.filterByArea = True
##params.minArea = 3.141 * 15**2#20000
##params.maxArea = 3.141 * 40**2#40000
##	 
### Filter by Circularity
##params.filterByCircularity = True
##params.minCircularity = 0.25
## 
### Filter by Convexity
####params.filterByConvexity = True
####params.minConvexity = 0.87
##	 
### Filter by Inertia
##params.filterByInertia = True
##params.minInertiaRatio = 0.8
##
### Distance Between Blobs
##params.minDistBetweenBlobs = 1000
##	 
### Create a detector with the parameters
##detector = cv2.SimpleBlobDetector_create(params)

    
# ************************************************************************
# =========================> MAKE IT ALL HAPPEN <=========================
# ************************************************************************
global lower_bound, upper_bound

##lower_bound = np.array( [0,0,10] )
##upper_bound = np.array( [255,255,195] )

lower_bound = np.array( [0,0,0] )
upper_bound = np.array( [10,10,10] )

while( True ):
    # Capture frame
    frame = stream.read()[36:252, 48:336]                                   # Capture frame and crop it
    image_ori = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)                      # Convert to grayscale
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
        
    # Display feed
    if( realDisplay ):                                                      # Show real output
        cv2.imshow( ver, image)                                             # 
        cv2.imshow( "CV", mask )                                            # ...
        key = cv2.waitKey(1) & 0xFF                                         # ...
    else:                                                                   # Show morphed image
        cv2.imshow(ver, closing)                                            # ...
        cv2.imshow( "CV", mask )                                            # ...
        key = cv2.waitKey(1) & 0xFF                                         # ...
        
