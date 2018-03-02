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
from    time                        import  sleep           # Sleep for stability
from    timeStamp                   import  fullStamp       # Show date/time on console output
import os, os.path

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
def procFrame( bgr2gray ):

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
    return( bgr2gray )


# ******************************************************
# Define a function to scan for circles from camera feed
# ******************************************************
def scan4circles( bgr2gray, frame ):

##    # Scan for circles
##    circles = cv2.HoughCircles( bgr2gray, cv2.HOUGH_GRADIENT, dp, minDist,
##                                param1, param2, minRadius, maxRadius )
##
    # Scan for circles
    circles = cv2.HoughCircles( bgr2gray, cv2.HOUGH_GRADIENT, 32, 455,
                                193, 191, 5, 40 )


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
        
        return( final )

    else:
        return( frame )


# ************************************************************************
# ===========================> SETUP PROGRAM <===========================
# ************************************************************************

# Setup camera
#img = cv2.imread( 'image.jpg' )
imageDir = "./"
image_path_list = []
valid_image_extensions = [".jpg", ".jpeg"]
valid_image_extensions = [item.lower() for item in valid_image_extensions]

for file in os.listdir(imageDir):
    extension = os.path.splitext(file)[1]
    if extension.lower() not in valid_image_extensions:
        continue
##    image_path_list.append(os.path.join(imageDir, file))
    image_path_list.append( file )
normalDisplay = True
sleep( 1.0 )

### Setup window and mouseCallback event
##cv2.namedWindow( ver )
##cv2.setMouseCallback( ver, control )
##
### Create a track bar for HoughCircles parameters
##cv2.createTrackbar( "dp"        , ver, 32   , 50 , placeholder ) #34
##cv2.createTrackbar( "minDist"   , ver, 455  , 750, placeholder )
##cv2.createTrackbar( "param1"    , ver, 193  , 750, placeholder ) #396
##cv2.createTrackbar( "param2"    , ver, 191  , 750, placeholder ) #236
##cv2.createTrackbar( "minRadius" , ver, 5    , 200, placeholder ) #7
##cv2.createTrackbar( "maxRadius" , ver, 40   , 250, placeholder )

# Setup window and trackbars for AI view
cv2.namedWindow( "AI_View" )

cv2.createTrackbar( "Type:\n0.Binary\n1.BinaryInv\n2.Trunc\n3.2_0\n4.2_0Inv",
                    "AI_View", 0, 4, placeholder )#3
cv2.createTrackbar( "thresholdVal", "AI_View", 63, 254, placeholder ) #30
cv2.createTrackbar( "maxValue", "AI_View", 255, 255, placeholder )#255


# ************************************************************************
# =========================> MAKE IT ALL HAPPEN <=========================
# ************************************************************************
for imagePath in image_path_list:
    print( "Reading image %s" %imagePath )
    img = cv2.imread( imagePath )
    cv2.imshow( 'window', img )
    frame = img
    output = frame

    # Convert into grayscale because HoughCircle only accepts grayscale images
    bgr2gray = cv2.cvtColor( frame, cv2.COLOR_RGB2GRAY )

    bgr2gray = procFrame( bgr2gray )

    # Get trackbar position and reflect it in HoughCircles parameters input
    dp = cv2.getTrackbarPos( "dp", ver )
    minDist = cv2.getTrackbarPos( "minDist", ver )
    param1 = cv2.getTrackbarPos( "param1", ver )
    param2 = cv2.getTrackbarPos( "param2", ver )
    minRadius = cv2.getTrackbarPos( "minRadius", ver )
    maxRadius = cv2.getTrackbarPos( "maxRadius", ver )

    # Start thread to scan for circles
    output = scan4circles( bgr2gray, frame )
    img_name = './done/[MODIFIED]%s' %imagePath
    print( img_name )
    cv2.imwrite( img_name, output )

cv2.destroyAllWindows()

### ************************************************************************
### ======================+> IMAGE BY IMAGE PROCESS <=======================
### ************************************************************************
##i=0
##while( True ):
##    img = cv2.imread( image_path_list[i] )
##    cv2.imshow( 'window', img )
##    frame = img
##    output = frame
##
##    # Convert into grayscale because HoughCircle only accepts grayscale images
##    bgr2gray = cv2.cvtColor( frame, cv2.COLOR_RGB2GRAY )
##
##    bgr2gray = procFrame( bgr2gray )
##
##    # Get trackbar position and reflect it in HoughCircles parameters input
##    dp = cv2.getTrackbarPos( "dp", ver )
##    minDist = cv2.getTrackbarPos( "minDist", ver )
##    param1 = cv2.getTrackbarPos( "param1", ver )
##    param2 = cv2.getTrackbarPos( "param2", ver )
##    minRadius = cv2.getTrackbarPos( "minRadius", ver )
##    maxRadius = cv2.getTrackbarPos( "maxRadius", ver )
##
##    # Start thread to scan for circles
##    output = scan4circles( bgr2gray, frame )
##
##    # Live feed display toggle
##    if normalDisplay:
##        cv2.imshow(ver, output)
##        cv2.imshow( "AI_View", bgr2gray )
##        key = cv2.waitKey(1) & 0xFF
##        if key == 27:
##            print( 'Next image' )
##            i=i+1
##            if i == 9: i=0
##    elif not(normalDisplay):
##        cv2.imshow(ver, bgr2gray)
##        key = cv2.waitKey(1) & 0xFF
##        if key == 27:
##            print( 'Next image' )
##            i=i+1
##            if i == 9: i=0
