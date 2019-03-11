import requests                                             # Allows you to send organic, grass-fed HTTP requests
import numpy                                    as np       # Always behind the scenes
import cv2                                                  # Runs the whole show
import os, platform                                         # Directory/file manipulation
from argparse       import ArgumentParser                   # Pass flags/parameters to script
import time
from    imutils.video.pivideostream     import  PiVideoStream
from picamera.array import PiRGBArray
from picamera import PiCamera

# ************************************************************************
# =====================> CONSTRUCT ARGUMENT PARSER <=====================*
# ************************************************************************

ap = ArgumentParser()

ap.add_argument( "-o", "--overlay", required=False,
                 help="Path to overlay image" )

ap.add_argument( "-a", "--alpha", type=float, default=0.85,
                 help="Set alpha level.\nDefault=0.85" )

ap.add_argument( "-d", "--debug", action='store_true',
                 help="Enable debugging" )

args = vars( ap.parse_args() )

# ************************************************************************
# =====================> DEFINE Camera Class <===========================*
# ************************************************************************

class Camera:
    
    def __init__(self, ip = '192.168.1.88', user = 'admin', pwd = 'admin'):                             # default values are camera defaults
        
        self.ip = str(ip)
        self.user = str(user)
        self.pwd = str(pwd)

    def PTZ(self, action, count = 1):                                           # sends an http request with url to complete action
        
        ip = self.ip
        user = self.user
        pwd = self.pwd                                                                                  # actions are ZoomAdd, ZoomSub, FocusAdd, and FocusSub
        action = str(action)
        request = 'http://{ip}/cgi-bin/ptz_cgi?action={action}&user={user}&pwd={pwd}'\
                  .format(ip = ip, action = action, user = user, pwd = pwd)
        
        for n in range(count):
            
            print(request)
            requests.get(request)

            
    def Stream(self):                                                           # OpenCv opens rtsp video stream
        
        ip = self.ip
        vcap = cv2.VideoCapture("rtsp://{ip}/av0_1".format(ip = ip))
        ret, frame = vcap.read()
        height, width = frame.shape[:2]
        
        print('resolution is {width}x{height}'.format(width=width, height=height))
        print(vcap.get(cv2.CAP_PROP_BUFFERSIZE))
        
        while(1):
            
            ret, frame = vcap.read()
            cv2.imshow('camera stream', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):                           # press keystroke "q" to break stream
                    
                break
                
        vcap.release()
        cv2.destroyAllWindows()
            
camera = Camera()

# ------------------------------------------------------------------------

def placeholder( x ):
    '''
    Place holder function for trackbars. This is required
    for the trackbars to function properly.
    '''
    
    return()

# ------------------------------------------------------------------------

def setup_detector():
    '''
    Setup blob detector.

    INPUTS:-
        - NONE

    OUTPUT:-
        - parameters: BLOB detector's parameters
        - detector  : BLOB detector
    '''
    
    parameters = cv2.SimpleBlobDetector_Params()                                # Parameters
	 
    # Filter by Area.
    parameters.filterByArea = True                                              # ...
    parameters.minArea = np.pi * 10**2                                          # ...
    parameters.maxArea = np.pi * 100**2                                          # ...

    # Filter by color
    parameters.filterByColor = True                                             # ...
    parameters.blobColor = 255                                                    # ...

    # Filter by Circularity
    parameters.filterByCircularity = True                                       # ...
    parameters.minCircularity = 0.2                                            # ...
     
    # Filter by Convexity
    parameters.filterByConvexity = False                                         # ...
##    parameters.minConvexity = 0.43                                              # ...
             
    # Filter by Inertia
    parameters.filterByInertia = True                                           # ...
    parameters.minInertiaRatio = 0.2                                          # ...

    # Distance Between Blobs
    parameters.minDistBetweenBlobs = 2000                                       # ...
             
    # Create a detector with the parameters
    detector = cv2.SimpleBlobDetector_create( parameters )                      # Create detector

    return( parameters, detector )
    
# ------------------------------------------------------------------------

def is_inROI( xyr_points=None, update_ROI=False ):
    '''
    Determine if we are in the ROI

    INPUTS:
        - xyr_points: (x, y, r) co-ordinates

    OUTPUT:
        - inRoi     : True/False boolean
    '''
    
    global ROI, startTime

    if( update_ROI ):                                                           # If update_ROI is true
        if( time() - startTime >= timeout ):                                    #   Check for timeout
            if( args["debug"] ): print( "[INFO] ROI Reset" )                    #       [INFO] status
            startTime = time()                                                  #       Reset timer
            return( ROI_0[:] )                                                  #       Reset ROI

        else:
            return( ROI )                                                       #   Else return current ROI

    else:
        x, y, r = xyr_points                                                    # Unpack co-ordinates
        
        x_min, x_max = x-r, x+r                                                 # Find min/max x-range
        y_min, y_max = y-r, y+r                                                 # Find min/max y-range

        ROI_x_min, ROI_y_min = ROI[0]                                           # Unpack min/max
        ROI_x_max, ROI_y_max = ROI[1]                                           # ROI points

        inROI = bool( False )                                                   # Boolean flag

        if( ROI_x_min < x_min and x_max < ROI_x_max ):                          # Check if we are within ROI
            if( ROI_y_min < y_min and y_max < ROI_y_max ):                      # ...
            
                ROI[0] = ( (x_min-dx), (y_min-dy) )                             #   Update ROI
                ROI[1] = ( (x_max+dx), (y_max+dy) )                             #   ...

                inROI  = bool( True )                                           #   Set flag to True
                
                startTime = time()                                              #   Reset timer
            
            else: pass                                                          # Else we are not...
        else: pass                                                              # within ROI

        return( inROI )

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
    hsv = cv2.cvtColor( frame, cv2.COLOR_BGR2HSV)
    minHSV = np.array([75, 10, 0])
    maxHSV = np.array([138, 255, 37])
    
    # Dissolve noise while maintaining edge sharpness
    processed = cv2.inRange(hsv, minHSV, maxHSV)
    processed = cv2.GaussianBlur( processed, (5, 5), 25 )
    
    return( processed )                                            # Return
    
# ------------------------------------------------------------------------

def prepare_overlay( img=None ):
    '''
    Load and prepare overlay images

    INPUTS:
        - img:     RAW   overlay iamge

    OUTPUT:
        - img: Processed overlay image
    '''
    
    # Check whether an overlay is specified
    if( img != None ):
        img = cv2.imread( img, cv2.IMREAD_UNCHANGED )                           # Load specific overlay
    else:
        src = "/home/pi/Desktop/video opthalomescope/Overlay.png"                               # Load default overlay
        img = cv2.imread( src  , cv2.IMREAD_UNCHANGED )                         # ...

    # Load overlay image with Alpha channel
    ( wH, wW ) = img.shape[:2]                                                  # Get dimensions
    ( B, G, R, A ) = cv2.split( img )                                           # Split into constituent channels
    B = cv2.bitwise_and( B, B, mask=A )                                         # Add the Alpha to the B channel
    G = cv2.bitwise_and( G, G, mask=A )                                         # Add the Alpha to the G channel
    R = cv2.bitwise_and( R, R, mask=A )                                         # Add the Alpha to the R channel
    img = cv2.merge( [B, G, R, A] )                                             # Finally, merge them back

    return( img )                                                               # Return processed overlay

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
    
    x, y, r = pos                                                               # Unpack co-ordinates
    
    x_min, x_max = x-r, x+r                                                     # Find min/max x-range
    y_min, y_max = y-r, y+r                                                     # Find min/max y-range

    if( x_min > 0 and y_min > 0 ):
        if( x_max < w and y_max < h ):
            
            overlay_img = cv2.resize( overlay_img, ( 2*r, 2*r ),                # Resize overlay image to fit
                                      interpolation = cv2.INTER_AREA )          # ...
            
            overlay_frame[ y_min:y_max, x_min:x_max] = overlay_img              # Place overlay image into overlay frame
            r_min       = 15                                                    # Get updated blob detector
            r_max       = 45                                                    # parameters
            alpha_val = np.interp( r, [r_min, r_max], [0.0, 1.0] )
            args["alpha"] = alpha_val
            frame = cv2.addWeighted( overlay_frame,                             # Join overlay frame (alpha)
                                     args["alpha"],                             # with actual frame (RGB)
                                     frame, 1.0, 0 )            # ...

            if( args["debug"] ):
                cv2.circle( frame, (x, y), r, (0, 255, 0), 2 )                  # Draw a circle
##                cv2.rectangle( frame, ROI[0], ROI[1], (255, 0, 0), 2 )          # Draw dynamic ROI rectangle
            
    return( frame )

# ------------------------------------------------------------------------

initC = True
initK = True

def find_pupil( processed, overlay_frame, overlay_img, frame ):
    '''
    Find pupil by scanning for circles within an image

    INPUTS:-
        - processed     : Processed image
        - overlay_frame : The overlay empty frame
        - overlay_img   : The overlay image
        - frame         : Frame to which we should attach overlay

    OUTPUT:-
        - frame         : Image with/without overlay
                (depends whether circles were found or not)
    '''
    global ROI
    global initC, initK
    
    ROI_x_min, ROI_y_min = ROI[0]
    ROI_x_max, ROI_y_max = ROI[1]
    
    frame_failed = bool( True )                                                 # Boolean flag 1
    within_ROI   = bool( False )                                                # Boolean flag 2

    # Error handling (1/3)
    try:
        keypoints = detector.detect( processed )                                     # Launch blob detector

        if( len(keypoints) > 0 ):                                               # If blobs are found
            if( args["debug"] ):
                if( initK ):
                    initC = True 
                    initK = False
                    print( "[INFO] cv2.SimpleBlobDetector()" )
                    
            for k in keypoints:                                                 # Iterate over found blobs
                x, y, r = int(k.pt[0]), int(k.pt[1]), int(k.size/2)             # Get co-ordinates
                pos = ( x, y, r )                                               # Pack co-ordinates
                
##            if( is_inROI( pos ) ):                                          # Check if we are within ROI
                frame = add_overlay( overlay_frame,                         # Add overlay
                                     overlay_img,                           # ...
                                     frame, pos )                           # ...
                    
##                    frame_failed = bool( False )                                # Set flag to false
##                    within_ROI   = bool( True  )                                # Set flag to true
                    
        else:
            # Contour Detector
            r_min   = 15                                                        # Get current r_min ...
            r_max   = 45                                                        # and r_max values
             
            _, contours, _ = cv2.findContours( processed,                       # Find contours based on their...
                                               cv2.RETR_EXTERNAL,               # external edges and keep only...
                                               cv2.CHAIN_APPROX_SIMPLE )        # intermediate points (SIMPLE)
            
            # If circles are found draw them
            if( len(contours) > 0 ):                                            # Check if we detected anything
                if( args["debug"] ):
                    if( initC ):
                        initC = False
                        initK = True                    
                        print( "[INFO] cv2.findContours()" )
                        
                for c in contours:                                              # Iterate over all contours
                    (x, y) ,r = cv2.minEnclosingCircle(c)                       # Min enclosing circle inscribing contour
                    xy = (int(x), int(y))                                       # Get circle's center...
                    r = int(r)                                                  # and radius
                    pos = ( xy[0], xy[1], r )                                   # Pack co-ordinates
                    
                    if( is_inROI( pos ) ):                                      # Check if we are within ROI
                        if( r_min <= r and r <= r_max ):                        # Check if within desired limit
                            frame = add_overlay( overlay_frame,                 # Add overlay
                                                 overlay_img,                   # ...
                                                 frame, pos )                   # ...

                            frame_failed = bool( False )                        # Set flag to false
                            within_ROI   = bool( True  )                        # Set flag to true


        if( frame_failed and not within_ROI ):
            ROI = is_inROI( update_ROI=True )                                   # Reset ROI if necessary

    # Error handling (2/3)
    except Exception as error:
        if( args["debug"] ):
            print( "{} Error caught in find_pupil()".format(FS()) )             # Specify error type
            print( "{0} {1}".format(FS(), error) )                              # ...


    # Error handling (3/3)
    finally:
        return( frame )                                                          

# ------------------------------------------------------------------------

realDisplay = True 

global ROI, lower_bound, upper_bound
dx, dy, dROI = 35, 35, 65
ROI_0 = [ (144-dROI, 108-dROI), (144+dROI, 108+dROI) ]
ROI   = [ (144-dROI, 108-dROI), (144+dROI, 108+dROI) ]
##lower_bound = np.array( [  0,   0,   0], dtype = np.uint8 )
##upper_bound = np.array( [180, 255,  30], dtype = np.uint8 )

global detector, params
params, detector = setup_detector() 

ip = camera.ip
##stream = cv2.VideoCapture("rtsp://{ip}/av0_1".format(ip = ip))
stream = cv2.VideoCapture('output.avi')

overlayImg  = prepare_overlay("/home/pi/Desktop/video_opthalomescope/Alpha/Leukemic-Retinopathy-1.png")

while(stream.isOpened()):
        
    ret, frame = stream.read()
    if ret == True:
        image = frame
        (h, w) = frame.shape[:2]                                                    # Determine width and height
        frame = np.dstack([ frame, np.ones((h, w),                                  # Stack the arrays in sequence, 
                                           dtype="uint8")*255 ])                    # depth-wise ( along the z-axis )

        overlay = np.zeros( ( h, w, 4 ), "uint8" )                                  # Empty np array w\ same dimensions as the frame

        proc = procFrame( image )


        image = find_pupil( proc, overlay, overlayImg, frame )

        if( realDisplay ):                                                          # Show real output
            cv2.imshow( 'Tracked Overlay', image)                                                 # 
            if cv2.waitKey(1) & 0xFF == ord('q'):                           # press keystroke "q" to break stream
                    
                break
        else:                                                                       # Show morphed image
            cv2.imshow( 'proccessed image', proc )                                              # ...
            if cv2.waitKey(1) & 0xFF == ord('q'):                           # press keystroke "q" to break stream
                    
                break
    else:
        break

stream.release()
cv2.destroyAllWindows()


    


















# !@#$%^&*()(*&^%$#@!!@#$%^*(*&(_(%^&$#%@!@#$%^&)(*&^%$#@!@#$%^&*()(*&^!@#

##pupilDetect()
##camera.Stream() 

###--------------------Picamera

##camera = PiCamera()
##camera.resolution = (640, 480)
##camera.framerate = 32
##rawCapture = PiRGBArray(camera, size=(640, 480))
##
##parameters = cv2.SimpleBlobDetector_Params()                                # Parameters
## 
### Filter by Area.
##parameters.filterByArea = True                                              # ...
##parameters.minArea = np.pi * 15**2                                          # ...
##parameters.maxArea = np.pi * 100**2                                         # ...
##
### Filter by color
##parameters.filterByColor = True                                             # ...
##parameters.blobColor = 0                                                    # ...
##
### Filter by Circularity
##parameters.filterByCircularity = True                                       # ...
##parameters.minCircularity = 0.42                                            # ...
## 
### Filter by Convexity
##parameters.filterByConvexity = True                                         # ...
##parameters.minConvexity = 0.43                                              # ...
##         
### Filter by Inertia
##parameters.filterByInertia = True                                           # ...
##parameters.minInertiaRatio = 0.41                                           # ...
##
### Distance Between Blobs
##parameters.minDistBetweenBlobs = 2000                                       # ...
##
##
##detector = cv2.SimpleBlobDetector_create( parameters )                      # Create detector
##
### allow the camera to warmup
##time.sleep(0.1)
##
### capture frames from the camera
##for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
##    # grab the raw NumPy array representing the image, then initialize the timestamp
##    # and occupied/unoccupied text
##    image = frame.array
##    keypoints = detector.detect(image)
##    img_detected = cv2.drawKeypoints(image, keypoints, np.array([]), (0,255,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
##    # show the frame
##    cv2.imshow("blob finder", img_detected)
##    key = cv2.waitKey(1) & 0xFF
##
##    # clear the stream in preparation for the next frame
##    rawCapture.truncate(0)
##
##    # if the `q` key was pressed, break from the loop
##    if key == ord("q"):
##        break
##    
##
##cv2.destroyAllWindows()





###---*----*
### ------------------------------------------------------------------------
##
##initC = True
##initK = True
##
##def pupilDetect():
##
##    ip = camera.ip
##    vcap = cv2.VideoCapture("rtsp://{ip}/av0_1".format(ip = ip))
##
##    
##    parameters = cv2.SimpleBlobDetector_Params()                                # Parameters
##     
##    # Filter by Area.
##    parameters.filterByArea = True                                              # ...
##    parameters.minArea = np.pi * 15**2                                          # ...
##    parameters.maxArea = np.pi * 100**2                                         # ...
##
##    # Filter by color
##    parameters.filterByColor = True                                             # ...
##    parameters.blobColor = 0                                                    # ...
##
##    # Filter by Circularity
##    parameters.filterByCircularity = True                                       # ...
##    parameters.minCircularity = 0.42                                            # ...
##     
##    # Filter by Convexity
##    parameters.filterByConvexity = True                                         # ...
##    parameters.minConvexity = 0.43                                              # ...
##             
##    # Filter by Inertia
##    parameters.filterByInertia = True                                           # ...
##    parameters.minInertiaRatio = 0.41                                           # ...
##
##    # Distance Between Blobs
##    parameters.minDistBetweenBlobs = 2000                                       # ...
##
##    
##    detector = cv2.SimpleBlobDetector_create( parameters )                      # Create detector
##
##
##
##    
##    
##    while(True):
##        ret, frame = vcap.read()
##        gray = cv2.cvtColor( frame, cv2.COLOR_BGR2GRAY )
##        keypoints = detector.detect( gray )
##        img_detected = cv2.drawKeypoints(frame, keypoints, np.array([]), (0,255,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
##        cv2.imshow('pupil detection', img_detected)
##        cv2.imshow('stream', frame)
##        cv2.imshow('processed', processed)
##
##            
##        if cv2.waitKey(1) & 0xFF == ord('q'):                           # press keystroke "q" to break stream
##                
##            break       
##
##    vcap.release()
##    cv2.destroyAllWindows()
##
### ------------------------------------------------------------------------
          




