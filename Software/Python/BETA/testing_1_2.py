import numpy                                    as np       # Always behind the scenes
import requests                                             # Allows the program to send organic, grass-fed HTTP requests
import cv2                                                  # It's always a good idea to give vision to computers
import json                                                 # Allows the program to write and read parameters for tracking 
import argparse                                             # Screw IDLE

#---------setting up argparse-----------

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--setvalues", action="store_true", help="allows you to set the program parameters")
args = parser.parse_args()

#---------setting up Camera class-----------

class Camera:
    ''' Camera Class for the IP Camera, has 3 main methods

    Pan-Tilt-Zoom (PTZ): allows the camera to zoom and focus using the associated action strings,
    *the computer Needs to have the same IP address as the camera to function properly*
    Actions:
        ZoomADD: zooms the camera in by one step then attempts autofocus;
        ZoomSub: zooms the camera out by one step then attempts autofocus;
        FocusADD: focus' the camera in by one step;
        FocusSub: focus' the camera out by one step;
        
    Stream: displays the camera stream, while printing the video resolution

    Record: records the video stream to file using the inputed file name (remember ".avi" at the end)
    '''
    def __init__(self, ip = '192.168.1.88', user = 'admin', pwd = 'admin'):  # Default values are camera defaults             
        self.ip = str(ip)
        self.user = str(user)
        self.pwd = str(pwd)
    def PTZ(self, action, count = 1):                                                
        ip = self.ip
        user = self.user
        pwd = self.pwd
        action = str(action)  # Actions are ZoomAdd, ZoomSub, FocusAdd, and FocusSub
        request = 'http://{ip}/cgi-bin/ptz_cgi?action={action}&user={user}&pwd={pwd}'\
                  .format(ip = ip, action = action, user = user, pwd = pwd)
        for n in range(count):  # Sends the request n (count) many times           
            print(request)
            requests.get(request)         
    def Stream(self):                                                                
        ip = self.ip
        vcap = cv2.VideoCapture("rtsp://{ip}/av0_1".format(ip = ip))
        ret, frame = vcap.read()
        height, width = frame.shape[:2]   # Finds resolution of video
        print('resolution is {width}x{height}'.format(width=width, height=height))      
        while(1):      
            ret, frame = vcap.read()
            cv2.imshow('camera stream', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):  # Press keystroke "q" to break stream                                    
                break  
        vcap.release()
        cv2.destroyAllWindows()
    def Record(self, name = 'output_1.avi'):
        ip = self.ip
        vcap = cv2.VideoCapture("rtsp://{ip}/av0_1".format(ip = ip))
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter( name, fourcc, 20, (640,480) )
        while(vcap.isOpened()):
            ret, frame = vcap.read()
            out.write(frame)
            cv2.imshow('frame',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print('Saving as {name}'.format(name = name))
                break
        # Release everything if job is finished
        vcap.release()
        out.release()
        cv2.destroyAllWindows()
            
camera = Camera()  # For easier access

#---------setting up Tracking and Overlaying Functions-----------

def placeholder( x ):
    # Needed for trackbar functinality
    pass
def placeholder2( x ):
    # Needed for trackbar functinality
    pass
def placeholder3( x ):
    # Needed for trackbar functinality
    pass
#-----

def setup_windows2():
    # Sets up overlay scaling window for final display
    cv2.namedWindow( "ScaleBar" )
    cv2.moveWindow("ScaleBar", 0, 0)  # Sets window postion to top left of display
    cv2.createTrackbar( "OverlayScaleValue", "ScaleBar", 100, 200, placeholder2 )
    return
def setup_windows():
    # Sets up overlay scaling window
    cv2.namedWindow( "ScaleBar" )
    cv2.moveWindow("ScaleBar", 0, 0)  # Sets window postion to top middle of display
    cv2.createTrackbar( "OverlayScaleValue", "ScaleBar", 100, 200, placeholder2 )
    # Setup blob detecting window
    cv2.namedWindow( "BlobParamsBar" )                                                  
    cv2.moveWindow("BlobParamsBar", 640, 0)
    # Trackbars for Blobdetector parameters
    cv2.createTrackbar( "Black = 0 White = 255"  , "BlobParamsBar", 255, 255, placeholder )
    cv2.createTrackbar( "minRadius"              , "BlobParamsBar", 15 , 100, placeholder )
    cv2.createTrackbar( "maxRadius"              , "BlobParamsBar", 300 , 500, placeholder )
    cv2.createTrackbar( "Circularity"            , "BlobParamsBar", 26 , 100, placeholder )
    cv2.createTrackbar( "Convexity"              , "BlobParamsBar", 43 , 100, placeholder )
    cv2.createTrackbar( "InertiaRatio"           , "BlobParamsBar", 41 , 100, placeholder )
    # Trackbars for HSV parameters
    cv2.namedWindow("HSVbars", 0)
    cv2.moveWindow("HSVbars", 1280, 0)  # Sets window postion to top right of display
    for i in ["MIN", "MAX"]:  # Creating Trackbars for HSV min and max
        v = 0 if i == "MIN" else 255
        for j in "HSV":
            cv2.createTrackbar("%s_%s" % (j, i), "HSVbars", v, 255, placeholder3)   
    return()

#-----

def ScaleBar_update():
    # Updates the ScaleBar value for functions
    value = cv2.getTrackbarPos( "OverlayScaleValue", "ScaleBar" )
    return( value )
def update_detector():
    # Updates Blobdetector parameters for functions
    global detector, parameters
    parameters = cv2.SimpleBlobDetector_Params()
    # Retrieves parameters from trackbars realtime
    BlackOrWhite = cv2.getTrackbarPos( "Black = 0 White = 255"  , "BlobParamsBar" )
    r_min        = cv2.getTrackbarPos( "minRadius"              , "BlobParamsBar" )
    r_max        = cv2.getTrackbarPos( "maxRadius"              , "BlobParamsBar" )
    circle_min   = cv2.getTrackbarPos( "Circularity"            , "BlobParamsBar" )
    convex_min   = cv2.getTrackbarPos( "Convexity"              , "BlobParamsBar" )
    inertia_min  = cv2.getTrackbarPos( "InertiaRatio"           , "BlobParamsBar" )
    # Filter by Area.
    parameters.filterByArea = True                                             
    parameters.minArea          = np.pi * r_min**2
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
    # Reflect changes
    detector = cv2.SimpleBlobDetector_create( parameters )                           
    return( parameters, detector )
def get_trackbar_values():
    # Gets the currect trackbar values and returns them
    values = []
    for i in ["MIN", "MAX"]:
        for j in "HSV":
            # Retrieves HSV values from trackbars
            v = cv2.getTrackbarPos("%s_%s" % (j, i), "HSVbars")
            values.append(v)
    return values

#-----

def setup_detector():                                                               
    # sets up detector with specified parameters for use in blob detector
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
    parameters.minConvexity = 0.1                                                       
    # Filter by Inertia
    parameters.filterByInertia = True                                     
    parameters.minInertiaRatio = 0.2                                          
    # Distance Between Blobs
    parameters.minDistBetweenBlobs = 2000                                     

    # Create detector
    detector = cv2.SimpleBlobDetector_create( parameters )

    return( parameters, detector )

#-----


def procFrame( image, minHSV=None, maxHSV=None ):
    '''procceses image so it is more reliable in blob detection
    takes in image; and possibly minHSV, and maxHSV values to proccess image
    if values are given it runs the proccessing, or else it takes HSV values
    from the running trackbars
    '''
    hsv = cv2.cvtColor( frame, cv2.COLOR_BGR2HSV)
    #switches to from RGB colorspace to HSV colorspace
    # min and max HSV values found from Range_Finder function
    if np.all(minHSV == None) and np.all(maxHSV == None):
        v1_min, v2_min, v3_min, v1_max, v2_max, v3_max = get_trackbar_values()
        minHSV = np.array([v1_min, v2_min, v3_min])
        maxHSV = np.array([v1_max, v2_max, v3_max])
        processed = cv2.inRange(hsv, minHSV, maxHSV)
        processed = cv2.GaussianBlur( processed, (5, 5), 25 )
    else:
        processed = cv2.inRange(hsv, minHSV, maxHSV)
    return( processed, minHSV, maxHSV )

#-----

def prepare_overlay( img=None ):
    # Check whether an overlay is specified
    if( img != None ):
        # Load specific overlay
        img = cv2.imread( img, cv2.IMREAD_UNCHANGED )                           
    else:
        src = "/home/pi/pd3d/repos/AugmentedOphthalmoscope/Software/Python/BETA/Alpha/Age-related macular degeneration.png"
        # Load default overlay
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
    
    # scale factor DO NOT SET TO ****ZERO**** bad things happen, like explosive things, bye bye program!!!
    
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
    else:
        detector = cv2.SimpleBlobDetector_create( parameters )
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
        'minCircularity' : parameters.minCircularity,
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
        minHSV = np.empty(3)
        for p in param_dict['minHSV']:
            minHSV[0] = p['minH']
            minHSV[1] = p['minS']
            minHSV[2] = p['minV']    
        maxHSV = np.empty(3)
        for p in param_dict['maxHSV']:
            maxHSV[0] = p['maxH']
            maxHSV[1] = p['maxS']
            maxHSV[2] = p['maxV']

    return ( parameters, minHSV, maxHSV )
                    
#---------Running the program-----------
    
##set_values = False

ip = camera.ip
##stream = cv2.VideoCapture("rtsp://{ip}/av0_1".format(ip = ip))
stream = cv2.VideoCapture("jack's_eye.avi")

overlayImg  = prepare_overlay("/home/pi/pd3d/repos/AugmentedOphthalmoscope/Software/Python/BETA/Alpha/Diabetic Retinopathy.png")


if(args.setvalues):
    setup_windows()
    params, detector = update_detector()
    while(stream.isOpened()):
        ret, frame = stream.read()
        if ret == True:
            image = frame
            (h, w) = frame.shape[:2]                                                    # Determine width and height
            frame = np.dstack([ frame, np.ones((h, w),                                  # Stack the arrays in sequence, 
                                               dtype="uint8")*255 ])                    # depth-wise ( along the z-axis )

            overlay = np.zeros( ( h, w, 4 ), "uint8" )                                  # Empty np array w\ same dimensions as the frame
            
            proc, minHSV, maxHSV = procFrame( image, None, None )


            image, img_detected, parameters = find_pupil( proc, overlay, overlayImg, frame )

            cv2.imshow( 'ScaleBar', image )                                                 # 
            cv2.imshow( 'BlobParamsBar', img_detected )
            cv2.imshow( 'HSVbars', proc)
            if cv2.waitKey(1) & 0xFF == ord('r'):                           # press keystroke "r" to reset stream
                write_parameters( minHSV, maxHSV, parameters )
                print("Resetting program now")
                break
        else:
            print("Stream over, resetting program now")
            break
        
    stream.release()
    cv2.destroyAllWindows()           
    # reset stream to reduce lag
##    stream = cv2.VideoCapture("rtsp://{ip}/av0_1".format(ip = ip))
    stream = cv2.VideoCapture("jack's_eye.avi")

    setup_windows2()
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

            cv2.imshow( 'ScaleBar', image )
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Closing program now")
                stream.release()
                cv2.destroyAllWindows()
        else:
            print("Stream over, closing program now")
            break


else:
    setup_windows2()
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

            cv2.imshow( 'ScaleBar', image )
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Closing program now")
                break
        else:
            print("Stream over, closing program now")
            break
            
stream.release()
cv2.destroyAllWindows()
            

          




