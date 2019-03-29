import numpy as np
import cv2

src = "/home/pi/pd3d/repos/AugmentedOphthalmoscope/Software/Python/BETA/Alpha/armd.jpeg"
# Load default overlay
img = cv2.imread( src  , cv2.IMREAD_UNCHANGED )
##stream = cv2.VideoCapture("lights_on.avi", cv2.WINDOW_NORMAL
##                          )

def placeholder(x):
    pass

def setup_windows():
    # Sets up overlay scaling window
    cv2.namedWindow( "Trackbar" )
    cv2.createTrackbar( "radius", "Trackbar", 450, 500, placeholder )
    cv2.createTrackbar( "x", "Trackbar", 640/2, 640, placeholder )
    cv2.createTrackbar( "y", "Trackbar", 480/2, 480, placeholder )

def ScaleBar_update():
    # Updates the ScaleBar value for functions
    radius = cv2.getTrackbarPos( "radius", "Trackbar" )
    x = cv2.getTrackbarPos( "x", "Trackbar" )
    y = cv2.getTrackbarPos( "y", "Trackbar" )
    return( radius, x, y )

setup_windows()
while(True):
    r, x, y = ScaleBar_update()
##    ret, frame = stream.read()
##    height,width,depth = frame.shape
##
##    ROI = frame[(y-r):(y+r),(x-r):(x+r)]
##    overlay = frame.copy()
##    output = frame.copy()
##    
##    cv2.circle(overlay,(x,y),radius, color=(0,0,0),thickness=-1)

    # apply the overlay
##    cv2.addWeighted(overlay, 0.5, output, 0.5,
##	    0, output)
    
    

    ( height, width ) = img.shape[:2]
    mask = np.zeros((height,width,4), np.uint8)
    black_image = np.zeros((height,width,4), np.uint8)
    r = 200
    cv2.circle(mask,(width/2,height/2),r,(255,255,255),-1,8,0)
    darken = cv2.bitwise_and( black_image, black_image, mask=mask )
##    mask = np.zeros((width,height),dtype=np.uint8)
##    cv2.circle(mask,(width/2,height/2),r,(255,255,255),-1,8,1)
##    # Split into constituent channels
##    ( B, G, R ) = cv2.split( img )
##    # Add the Alpha to the B channel
##    B = cv2.bitwise_and( B, B, mask=mask )
##    # Add the Alpha to the G channel
##    G = cv2.bitwise_and( G, G, mask=mask )
##    # Add the Alpha to the R channel
##    R = cv2.bitwise_and( R, R, mask=mask )
##    # Finally, merge them back
##    img = cv2.merge( [B, G, R] )                                             


##    img = cv2.bitwise_and( img, img, mask=mask )

    cv2.imshow('Trackbar', darken
               )
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
##
##stream.release()
cv2.destroyAllWindows()
            
