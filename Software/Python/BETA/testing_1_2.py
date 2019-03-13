import requests                                             # Allows you to send organic, grass-fed HTTP requests
import numpy                                    as np       # Always behind the scenes
import cv2                                                  # It's always a good idea to give vision to computers
import json

# ************************************************************************
# ========================> DEFINE Camera Class <========================*
# ************************************************************************

class Camera:
    
    def __init__(self, ip = '192.168.1.88', user = 'admin', pwd = 'admin'):          # default values are camera defaults
        
        self.ip = str(ip)
        self.user = str(user)
        self.pwd = str(pwd)

    def PTZ(self, action, count = 1):                                                # sends an http request with url to complete action
        
        ip = self.ip
        user = self.user
        pwd = self.pwd
        # actions are ZoomAdd, ZoomSub, FocusAdd, and FocusSub
        action = str(action)
        request = 'http://{ip}/cgi-bin/ptz_cgi?action={action}&user={user}&pwd={pwd}'\
                  .format(ip = ip, action = action, user = user, pwd = pwd)
        
        for n in range(count):
            
            print(request)
            requests.get(request)

            
    def Stream(self):                                                                # OpenCv opens rtsp video stream
        
        ip = self.ip
        vcap = cv2.VideoCapture("rtsp://{ip}/av0_1".format(ip = ip))
        ret, frame = vcap.read()
        # finds resolution of video
        height, width = frame.shape[:2] 
        print('resolution is {width}x{height}'.format(width=width, height=height))      
        while(1):
            
            ret, frame = vcap.read()
            cv2.imshow('camera stream', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):                                    # press keystroke "q" to break stream
                    
                break
                
        vcap.release()
        cv2.destroyAllWindows()

    def Record(self, name = 'output.avi'):
        
        ip = self.ip
        vcap = cv2.VideoCapture("rtsp://{ip}/av0_1".format(ip = ip))
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter( name, fourcc, 20, (640,480) )

        while(vcap.isOpened()):
            ret, frame = vcap.read()
            out.write(frame)
            cv2.imshow('frame',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release everything if job is finished
        vcap.release()
        out.release()
        cv2.destroyAllWindows()
            
camera = Camera()

### ************************************************************************
### ===================> DEFINE Range_Finder Function <====================*
### ************************************************************************
##
### open the camera
##ip = camera.ip
##cap = cv2.VideoCapture("rtsp://{ip}/av0_1".format(ip = ip))
##
####cap = cv2.VideoCapture('output.avi')
##
### range type
##range_filter = "HSV"
##
##def callback(value):                                                                 # blank function for trackbar functionality
##    pass
##
##def setup_trackbars(range_filter):                                                   # sets up trackbars for HSV and initial values
##    cv2.namedWindow("Trackbars", 0)
##
##    for i in ["MIN", "MAX"]:
##        v = 0 if i == "MIN" else 255
##
##        for j in range_filter:
##            cv2.createTrackbar("%s_%s" % (j, i), "Trackbars", v, 255, callback)
##
##def get_trackbar_values(range_filter):                                               # gets the currect trackbar values and returns them
##    values = []
##
##    for i in ["MIN", "MAX"]:
##        for j in range_filter:
##            v = cv2.getTrackbarPos("%s_%s" % (j, i), "Trackbars")
##            values.append(v)
##
##    return values
##
##def Range_Finder():                                                                  # Set the HSV Min and Max values through trackbars, outputs thresholded image
##    setup_trackbars(range_filter)
##     
##    while True:
##
##        #read the image from the camera
##        ret, frame = cap.read()
##        if ret == True:
##            # converting to HSV
##            hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
##            # get info from track bar and appy to result
##            v1_min, v2_min, v3_min, v1_max, v2_max, v3_max = get_trackbar_values(range_filter)
##            # threshholds image with inrange function
##            thresh = cv2.inRange(hsv, (v1_min, v2_min, v3_min), (v1_max, v2_max, v3_max))
##            # converts back to RGB with mask applied
##            result = cv2.bitwise_and(frame,frame,mask = thresh)
##
##            # video windows of original, thresholded, and result
####            cv2.imshow("Original", frame)
##            cv2.imshow('result', result)
##            cv2.imshow("Thresh", thresh)
##
##            # Quit the program when Q is pressed and prints HSV Min and Max values
##            if cv2.waitKey(1) & 0xFF == ord('q'):
##                print("minHSV", get_trackbar_values(range_filter)[:3])
##                print("maxHSV", get_trackbar_values(range_filter)[3:])
##                break
##        # Quits the program if end of video is reached and prints HSV Min and Max values
##        else:
##            print("minHSV", get_trackbar_values(range_filter)[:3])
##            print("maxHSV", get_trackbar_values(range_filter)[3:])
##            break
##
##    # When everything done, release the capture
##    print 'closing program' 
##    cap.release()
##    cv2.destroyAllWindows()

# ************************************************************************
# ===============> DEFINE Tracking and Overlaying Functions <============*
# ************************************************************************

def placeholder( x ):
    
    #needed for trackbar functinality
    pass

#-----

def placeholder2( x ):
    
    #needed for trackbar functinality
    pass

#-----

def placeholder3( x ):
    
    #needed for trackbar functinality
    pass

#-----

def setup_windows():
    
    # sets up overlay scaling window
    cv2.namedWindow( "ScaleBar" )
    cv2.createTrackbar( "OverlayScaleValue", "ScaleBar", 100, 200, placeholder2 )
    
    # Setup blob detecting window
    cv2.namedWindow( "BlobParamsBar" )                                                  # Start a named window for output
    cv2.createTrackbar( "Black = 0 White = 255"  , "BlobParamsBar", 255, 255, placeholder )
    cv2.createTrackbar( "minRadius"              , "BlobParamsBar", 15 , 100, placeholder )      # Trackbars for blob detector
    cv2.createTrackbar( "maxRadius"              , "BlobParamsBar", 300 , 500, placeholder )      # parameters.
    cv2.createTrackbar( "Circularity"            , "BlobParamsBar", 26 , 100, placeholder )#40   #     (used in find_pupil)
    cv2.createTrackbar( "Convexity"              , "BlobParamsBar", 43 , 100, placeholder )#15   # ...
    cv2.createTrackbar( "InertiaRatio"           , "BlobParamsBar", 41 , 100, placeholder )      # ...

    cv2.namedWindow("HSVbars", 0)
    for i in ["MIN", "MAX"]:
        v = 0 if i == "MIN" else 255

        for j in "HSV":
            cv2.createTrackbar("%s_%s" % (j, i), "HSVbars", v, 255, placeholder3)
            
    return()

#-----
def ScaleBar_update():
    
    value = cv2.getTrackbarPos( "OverlayScaleValue", "ScaleBar" )

    return( value )

#-----

def update_detector():

    global detector, parameters
    
    parameters = cv2.SimpleBlobDetector_Params()

    
    BlackOrWhite = cv2.getTrackbarPos( "Black = 0 White = 255"  , "BlobParamsBar" )
    r_min        = cv2.getTrackbarPos( "minRadius"              , "BlobParamsBar" )                    # Get updated blob detector
    r_max        = cv2.getTrackbarPos( "maxRadius"              , "BlobParamsBar" )                    # parameters
    circle_min   = cv2.getTrackbarPos( "Circularity"            , "BlobParamsBar" )                    # ...
    convex_min   = cv2.getTrackbarPos( "Convexity"              , "BlobParamsBar" )                    # ...
    inertia_min  = cv2.getTrackbarPos( "InertiaRatio"           , "BlobParamsBar" )                    # ...
    
    # Filter by Area.
    parameters.filterByArea = True                                             
    parameters.minArea          = np.pi * r_min**2                                  # Update parameters
    parameters.maxArea          = np.pi * r_max**2                         
    # Filter by color
    parameters.filterByColor = True                                
    parameters.blobColor = BlackOrWhite                                       
    # Filter by Circularity
    parameters.filterByCircularity = True                             
    parameters.minCircularity   = circle_min/100.                                            
    # Filter by Convexity
    parameters.filterByConvexity = True                                       
    parameters.minConvexity     = convex_min/100                                                       
    # Filter by Inertia
    parameters.filterByInertia = True                                     
    parameters.minInertiaRatio  = inertia_min/100                                          
    # Distance Between Blobs
    parameters.minDistBetweenBlobs = 2000

    
    detector = cv2.SimpleBlobDetector_create( parameters )                          # Reflect changes 

    return( parameters, detector )
#-----

def get_trackbar_values():                                               # gets the currect trackbar values and returns them
    values = []

    for i in ["MIN", "MAX"]:
        for j in "HSV":
            v = cv2.getTrackbarPos("%s_%s" % (j, i), "HSVbars")
            values.append(v)

    return values

#-----

def setup_detector():                                                               # sets up detector with specified parameters for use in blob detector
    
    # Parameters for blob detector
    parameters = cv2.SimpleBlobDetector_Params()                                          
    # Filter by Area.
    parameters.filterByArea = True                                             
    parameters.minArea = np.pi * 15**2                        
    # Filter by color
    parameters.filterByColor = True                                
    parameters.blobColor = 255                                       
    # Filter by Circularity
    parameters.filterByCircularity = True                             
    parameters.minCircularity = .2                                            
    # Filter by Convexity
    parameters.filterByConvexity = False                                       
##            parameters.minConvexity = 0.1                                                       
    # Filter by Inertia
    parameters.filterByInertia = True                                     
    parameters.minInertiaRatio = 0.2                                          
    # Distance Between Blobs
    parameters.minDistBetweenBlobs = 2000                                     

    # Create detector
    detector = cv2.SimpleBlobDetector_create( parameters )

    return( parameters, detector )                                                  # returns parameters and detector

###-----
##
##def Range_Finder( image ):                                                                  # Set the HSV Min and Max values through trackbars, outputs thresholded image
##
##    # converting to HSV
##    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
##    # get info from track bar and appy to result
##    v1_min, v2_min, v3_min, v1_max, v2_max, v3_max = get_trackbar_values()
##    minHSV = np.array([v1_min, v2_min, v3_min])
##    maxHSV = np.array([v1_max, v2_max, v3_max])
##    # threshholds image with inrange function
##    thresh = cv2.inRange(hsv, (v1_min, v2_min, v3_min), (v1_max, v2_max, v3_max))
##
##    return (thresh, minHSV, maxHSV )

#-----

def procFrame( image, minHSV=None, maxHSV=None ):                                                             # procceses image so it is more reliable in blob detection
    #switches to from RGB colorspace to HSV colorspace
    hsv = cv2.cvtColor( frame, cv2.COLOR_BGR2HSV)
    # min and max HSV values found from Range_Finder function
    if minHSV and maxHSV == None:
        v1_min, v2_min, v3_min, v1_max, v2_max, v3_max = get_trackbar_values()
##        minHSV = np.array([75, 10, 0])
##        maxHSV = np.array([138, 255, 37])
##        minHSV = np.array([0, 54, 0])
##        maxHSV = np.array([255, 255, 255])
        minHSV = np.array([v1_min, v2_min, v3_min])
        maxHSV = np.array([v1_max, v2_max, v3_max])
    else:
        pass
    # Dissolve noise while maintaining edge sharpness
##    processed = cv2.inRange(hsv, (v1_min, v2_min, v3_min), (v1_max, v2_max, v3_max))
    processed = cv2.inRange(hsv, minHSV, maxHSV)
    processed = cv2.GaussianBlur( processed, (5, 5), 25 )
    
    return( processed, minHSV, maxHSV )                                                             # returns processed image

#-----

def prepare_overlay( img=None ):
    
    # Check whether an overlay is specified
    if( img != None ):
        # Load specific overlay
        img = cv2.imread( img, cv2.IMREAD_UNCHANGED )                           
    else:
        # Load default overlay
        src = "/home/pi/pd3d/repos/AugmentedOphthalmoscope/Software/Python/BETA/Alpha/Age-related macular degeneration.png"                               
        img = cv2.imread( src  , cv2.IMREAD_UNCHANGED )                        

    # Load overlay image with Alpha channel
    
    # Get dimensions
    ( wH, wW ) = img.shape[:2]
    # Split into constituent channels
    ( B, G, R, A ) = cv2.split( img )
    # Add the Alpha to the B channel
    B = cv2.bitwise_and( B, B, mask=A )
    # Add the Alpha to the G channel
    G = cv2.bitwise_and( G, G, mask=A )
    # Add the Alpha to the R channel
    R = cv2.bitwise_and( R, R, mask=A )
    # Finally, merge them back
    img = cv2.merge( [B, G, R, A] )                                             

    return( img )                                                                   # Return processed overlay

#-----

def add_overlay( overlay_frame, overlay_img, frame, pos ):                          # scales, adds alpha weight, and overlays onto image

    # Unpack co-ordinates
    x, y, r = pos
    # scale factor DO NOT SET TO ****ZERO**** bad things happen
    scal_val = ScaleBar_update()
    r_scaled = r * scal_val / 100
    width = 2 * r_scaled
    height = 2 * r_scaled
    dim = ( width, height )
    
    # Find min/max x-range
    x_min, x_max = x-r_scaled, x+r_scaled
    # Find min/max y-range
    y_min, y_max = y-r_scaled, y+r_scaled                                                     

    if( x_min > 0 and y_min > 0 ):
        if( x_max < w and y_max < h ):
            # Resize overlay image to fit
            overlay_img = cv2.resize( overlay_img,  dim,
                                      interpolation=cv2.INTER_AREA )          
            # Place overlay image into overlay frame
            overlay_frame[ y_min:y_max, x_min:x_max] = overlay_img              

            # Join overlay frame (alpha) with actual frame (RGB)
            frame = cv2.addWeighted( overlay_frame,                             
                                     0.5,                              
                                     frame, 1.0, 0 )            

    return( frame )                                                                  # returns the frame with overlay added

#-----

def find_pupil( processed, overlay_frame, overlay_img, frame, parameters=None ):

    # Launch blob detector
    if parameters == None:
        parameters, detector = update_detector()
    keypoints = detector.detect( processed )
    img_detected = cv2.drawKeypoints(processed, keypoints,
                                             np.array([]), (0,255,0),
                                             cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # If blobs are found
    if( len(keypoints) > 0 ):
        # Iterate over found blobs
        for k in keypoints:
            # Get co-ordinates
            x, y, r = int(k.pt[0]), int(k.pt[1]), int(k.size/2)
            # Pack co-ordinates
            pos = ( x, y, r )
            # Add overlay
            frame = add_overlay( overlay_frame,                         
                                     overlay_img,                           
                                     frame, pos )

    return( frame, img_detected, parameters )

#-----

def write_parameters( minHSV, maxHSV, parameters ):
    param_dict = {}
    param_dict['minHSV'] = []
    param_dict['minHSV'].append({
        'minH' : minHSV[0],
        'minS' : minHSV[1],
        'minV' : minHSV[2]
    })
    param_dict['maxHSV'] = []
    param_dict['maxHSV'].append({
        'maxH' : maxHSV[0],
        'maxS' : maxHSV[1],
        'maxV' : maxHSV[2]
    })
    param_dict['blobparam'] = []
    param_dict['blobparam'].append({
        'minArea' : parameters.minArea,
        'maxArea' : parameters.maxArea,
        'color' : parameters.blobColor,
        'minCirularity' : parameters.minCircularity,
        'minConvexity' : parameters.minConvexity,
        'minInertiaRatio' : parameters.minInertiaRatio,
        'minDistBetweenBlobs' : parameters.minDistBetweenBlobs
    })
    with open( 'parameter_data', 'w' ) as outfile:
        json.dump( param_dict, outfile, indent=4 )

#-----

def read_parameters( filename='parameter_data' ):
    parameters = cv2.SimpleBlobDetector_Params()
    with open( filename ) as json_file:   
        param_dict = json.load(json_file)
        for p in param_dict['blobparam']:
            parameters.minDistBetweenBlobs = p['minDistBetweenBlobs']
            parameters.minInertiaRatio = p['minInertiaRatio']
            parameters.minConvexity = p['minConvexity']
            parameters.minCircularity = p['minCircularity']
            parameters.blobColor = p['color']
            parameters.maxArea = p['maxArea']
            parameters.minArea =  p['minArea']
        minHSV = []
        for p in param_dict['minHSV']:
            minHSV.append( p['minH'])
            minHSV.append( p['minS'])
            minHSV.append( p['minV'])     
        maxHSV = []
        for p in param_dict['minHSV']:
            maxHSV.append( p['maxH'])
            maxHSV.append( p['maxS'])
            maxHSV.append( p['maxV'])

    return ( parameters, minHSV, maxHSV )
        
#-----            
            
# standalone function to test pupil detection    
def pupil_detect():                                                                  # uses opencv blob detection to track pupil
    # video capture source    
    ip = camera.ip
    cap = cv2.VideoCapture("rtsp://{ip}/av0_1".format(ip = ip))

    setup_windows()

##  cap = cv2.VideoCapture('output.avi')
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            
            # Switch image from BGR colorspace to HSV
            hsv = cv2.cvtColor( frame, cv2.COLOR_BGR2HSV)
            # min and max HSV values found from Range_Finder function
##            minHSV = np.array([75, 10, 0])
##            maxHSV = np.array([138, 255, 37])
            minHSV = np.array([0, 54, 0])
            maxHSV = np.array([255, 255, 255])
            # Sets desired pixels to white, all else will be set to black
            mask = cv2.inRange(hsv, minHSV, maxHSV)
            # Blur image to remove noise
            mask = cv2.GaussianBlur(mask, (5, 5), 0)

            
            
            # Parameters for blob detector
            parameters = cv2.SimpleBlobDetector_Params()                                          
            # Filter by Area.
            parameters.filterByArea = True                                             
            parameters.minArea = np.pi * 15**2                        
            # Filter by color
            parameters.filterByColor = True                                
            parameters.blobColor = 255                                       
            # Filter by Circularity
            parameters.filterByCircularity = True                             
            parameters.minCircularity = .2                                            
            # Filter by Convexity
            parameters.filterByConvexity = False                                       
##            parameters.minConvexity = 0.1                                                       
            # Filter by Inertia
            parameters.filterByInertia = True                                     
            parameters.minInertiaRatio = 0.2                                          
            # Distance Between Blobs
            parameters.minDistBetweenBlobs = 2000

            update_detector()

##            # Create detector
##            detector = cv2.SimpleBlobDetector_create( parameters )
            

            
            # positional values from detector
            keypoints = detector.detect(mask)
            # displays tracked region over chosen image (first argument)
            img_detected = cv2.drawKeypoints(mask, keypoints,
                                             np.array([]), (0,255,0),
                                             cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            cv2.imshow('blob', img_detected)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()

#--------------------------------------------------

##Range_Finder() 
##pupil_detect()

#--------------------------------------------------
    
set_values = True 

ip = camera.ip
stream = cv2.VideoCapture("rtsp://{ip}/av0_1".format(ip = ip))
##stream = cv2.VideoCapture('output.avi')

overlayImg  = prepare_overlay("/home/pi/pd3d/repos/AugmentedOphthalmoscope/Software/Python/BETA/Alpha/Diabetic Retinopathy.png")

setup_windows()
##params, detector = update_detector()

while(stream.isOpened()):


        
    ret, frame = stream.read()
    if ret == True:
        image = frame
        (h, w) = frame.shape[:2]                                                    # Determine width and height
        frame = np.dstack([ frame, np.ones((h, w),                                  # Stack the arrays in sequence, 
                                           dtype="uint8")*255 ])                    # depth-wise ( along the z-axis )

        overlay = np.zeros( ( h, w, 4 ), "uint8" )                                  # Empty np array w\ same dimensions as the frame
        
        proc, minHSV, maxHSV = procFrame( image )


        image, img_detected, parameters = find_pupil( proc, overlay, overlayImg, frame )

        if( set_values ):                                                          # Show real output
            cv2.imshow( 'ScaleBar', image )                                                 # 
            cv2.imshow( 'BlobParamsBar', img_detected )
            cv2.imshow( 'HSVbars', proc)                               # ...
            if cv2.waitKey(1) & 0xFF == ord('q'):                           # press keystroke "q" to break stream  
                write_parameters( minHSV, maxHSV, parameters )
                break
        else:                                                                       # Show morphed image
            cv2.imshow( 'Stream with tracked overlay', image )
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    else:
        break

stream.release()
cv2.destroyAllWindows()

if( set_values ):
    while(stream.isOpened()):

        ret, frame = stream.read()
        if ret == True:
            image = frame
            (h, w) = frame.shape[:2]                                                    # Determine width and height
            frame = np.dstack([ frame, np.ones((h, w),                                  # Stack the arrays in sequence, 
                                               dtype="uint8")*255 ])                    # depth-wise ( along the z-axis )

            overlay = np.zeros( ( h, w, 4 ), "uint8" )                                  # Empty np array w\ same dimensions as the frame

            parameters, minHSV, maxHSV = read_parameters()
            
            proc, minHSV, maxHSV = procFrame( image, minHSV, maxHSV )


            image, img_detected, parameters = find_pupil( proc, overlay, overlayImg, frame, parameters )

            cv2.imshow( 'Stream with tracked overlay', image )

            

          




