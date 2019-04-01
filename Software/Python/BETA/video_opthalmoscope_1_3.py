import numpy                                    as np       # Always behind the scenes
import requests                                             # Allows the program to send organic, grass-fed HTTP requests
import cv2                                                  # It's always a good idea to give vision to computers
import json                                                 # Allows the program to write and read parameters for tracking 
import argparse                                             # CMD FTW
import os, platform



''' try grayscale
    try reducing ROI on processing image 100x100 vs 640x480
    try overlaying center crosshair 
'''

#---------setting up argparse-----------

parser = argparse.ArgumentParser()
parser.add_argument("-v", "--setvalues", action="store_true", help="allows you to set the program parameters")
parser.add_argument("-s", "--stream", default="rtsp://192.168.1.88/av0_1", type=str, help="sets the path to the\
image file, if none given goes to stream")
parser.add_argument("-o", "--overlay", required=False, type=str, help="sets the path to desired\
overlay image")
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

def control( event, x, y, flags, param ):
    '''
    Left click mouse events for overlay switching
    '''
    global overlayImg, counter
     
    # Left button switches overlays
    if( event == cv2.EVENT_LBUTTONDOWN ):                                       # If left-click, switch overlays
        print( "[INFO] Loading {}".format(overlay_name_list[counter]) ) ,       
        overlayImg = prepare_overlay( overlay_path_list[counter] )              # Switch overlay
        print( "...DONE" )                                                      
        counter += 1
        
        if( counter == len(overlay_path_list) ):
            counter = 0

def placeholder( x ):
    ''' Needed for trackbar functinality'''
    pass
def placeholder2( x ):
    ''' Needed for trackbar functinality'''
    pass
def placeholder3( x ):
    ''' Needed for trackbar functinality'''
    pass

#-----

def setup_windows2():
    ''' Sets up overlay scaling window for final display'''
    cv2.namedWindow( "ScaleBar" )
    cv2.moveWindow("ScaleBar", 0, 0)  # Sets window postion to top left of display
    cv2.createTrackbar( "OverlayScaleValue", "ScaleBar", 100, 200, placeholder2 )
    cv2.createTrackbar( "x-center", "ScaleBar", 320, 640, placeholder2 )
    cv2.createTrackbar( "y-center", "ScaleBar", 240, 480, placeholder2 )
    cv2.createTrackbar( "Zoom", "ScaleBar", 100, 200, placeholder2 )
    cv2.setMouseCallback( "ScaleBar", control )

    return
def setup_windows():
    ''' Sets up all main windows for value tuning'''
    # Sets up overlay scaling window
    cv2.namedWindow( "ScaleBar" )
    cv2.moveWindow("ScaleBar", 0, 0)  # Sets window postion to top middle of display
    cv2.createTrackbar( "OverlayScaleValue", "ScaleBar", 100, 200, placeholder2 )
    cv2.createTrackbar( "x-center", "ScaleBar", 320, 640, placeholder2 )
    cv2.createTrackbar( "y-center", "ScaleBar", 240, 480, placeholder2 )
    cv2.createTrackbar( "Zoom", "ScaleBar", 100, 200, placeholder2 )
    cv2.setMouseCallback( "ScaleBar", control )
    # Setup blob detecting window
    cv2.namedWindow( "BlobParamsBar" )                                                  
    cv2.moveWindow("BlobParamsBar", 640, 0)
    # Trackbars for Blobdetector parameters
    cv2.createTrackbar( "Black = 0 White = 255"  , "BlobParamsBar", 255, 255, placeholder )
    cv2.createTrackbar( "minRadius"              , "BlobParamsBar", 30 , 100, placeholder )
    cv2.createTrackbar( "maxRadius"              , "BlobParamsBar", 45 , 100, placeholder )
    cv2.createTrackbar( "Circularity"            , "BlobParamsBar", 50 , 100, placeholder )
    cv2.createTrackbar( "Convexity"              , "BlobParamsBar", 50 , 100, placeholder )
    cv2.createTrackbar( "InertiaRatio"           , "BlobParamsBar", 50 , 100, placeholder )
    # Trackbars for HSV parameters
    cv2.namedWindow("HSVbars", 0)
    cv2.moveWindow("HSVbars", 1280, 0)  # Sets window postion to top right of display
    for i in ["MIN", "MAX"]:  # Creating Trackbars for HSV min and max
        v = 0 if i == "MIN" else 255
        for j in "HSV":
            cv2.createTrackbar("%s_%s" % (j, i), "HSVbars", v, 255, placeholder3)   
    return()

#-----
                        
def Center_update():
    x_cent = cv2.getTrackbarPos( "x-center", "ScaleBar" )
    y_cent = cv2.getTrackbarPos( "y-center", "ScaleBar" )
    d = cv2.getTrackbarPos( "Zoom", "ScaleBar" )

    return( x_cent, y_cent, d)
def ScaleBar_update():
    ''' Updates the ScaleBar value for functions'''
    scale_val = cv2.getTrackbarPos( "OverlayScaleValue", "ScaleBar" )
    return( scale_val )
def update_detector():
    ''' Updates Blobdetector parameters for functions'''
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
    ''' Gets the currect trackbar values and returns them'''
    values = []
    for i in ["MIN", "MAX"]:
        for j in "HSV":
            # Retrieves HSV values from trackbars
            v = cv2.getTrackbarPos("%s_%s" % (j, i), "HSVbars")
            values.append(v)
    return values

#-----

def setup_detector():                                                               
    ''' Sets up detector with specified parameters for use in blob detector'''
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
    ''' Procceses image so it is more reliable in blob detection
    takes in image; and possibly minHSV, and maxHSV values to proccess image
    if values are given it runs the proccessing, or else it takes HSV values
    from the running trackbars
    '''
    height,width,depth = image.shape
    x, y, d = Center_update()
    mask = np.zeros((height,width), dtype="uint8")
    cv2.circle(mask,(x,y),240,(255,255,255),thickness=-1)
    ROI = cv2.bitwise_and( image, image, mask=mask )
    # Roi for the detecting function (faster and more reliable)
    ROI = cv2.cvtColor( ROI, cv2.COLOR_BGR2Lab)
    # Switches to from RGB colorspace to LAB colorspace
    # Min and max HSV values found from Range_Finder function
    if np.all(minHSV == None) and np.all(maxHSV == None):
        v1_min, v2_min, v3_min, v1_max, v2_max, v3_max = get_trackbar_values()
        minHSV = np.array([v1_min, v2_min, v3_min])
        maxHSV = np.array([v1_max, v2_max, v3_max])
        processed = cv2.inRange(ROI, minHSV, maxHSV)
        processed = cv2.GaussianBlur( processed, (5, 5), 25 )
    else:
        processed = cv2.inRange(ROI, minHSV, maxHSV)
        processed = cv2.GaussianBlur( processed, (5, 5), 25 )
    return( processed, minHSV, maxHSV )

#-----

def prepare_overlay( img=None ):
    ''' Prepares the overlay with an alpha channel'''
    # Check whether an overlay is specified
    if( img != None ):
        # Load specific overlay
        img = cv2.imread( img, cv2.IMREAD_UNCHANGED )                           
    else:
        src = "/home/pi/pd3d/repos/AugmentedOphthalmoscope/Software/Python/BETA/Alpha/Age-related macular degeneration.png"
        # Load default overlay
        img = cv2.imread( src  , cv2.IMREAD_UNCHANGED )
        # Load overlay image with Alpha channel
    
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


    return( img )                                           

#-----

def add_overlay( overlay_frame, overlay_img, frame, pos ):
    ''' Scales, adds alpha weight, and overlays onto image'''

    # Unpack co-ordinates
    x, y, r = pos
    
    # Scale factor DO NOT SET TO ****ZERO**** bad things happen, like explosive things, bye bye program!!!
    scale_val = ScaleBar_update()
    r_scaled = r * scale_val / 100
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
            overlay_img = cv2.resize( overlay_img, dim, interpolation=cv2.INTER_AREA )
            dark_circle = frame.copy()
            cv2.circle(dark_circle,(x,y),r_scaled-4, color=(0,0,0),thickness=-1)
            dark_circle = cv2.GaussianBlur( dark_circle, (5, 5), 25 )
            cv2.addWeighted(frame, 0.3, dark_circle, 0.7, 0, frame)
            
            # Place overlay image into overlay frame
            overlay_frame[ y_min:y_max, x_min:x_max] = overlay_img              

            # Join overlay frame (alpha) with actual frame (RGB)
            frame = cv2.addWeighted( overlay_frame,                             
                                     1,                              
                                     frame, 1, 0 )

    return( frame )

#-----

def find_pupil( processed, overlay_frame, overlay_img, frame, parameters=None ):
    ''' Finds keypoints using parameter data and blob detector for positional
    data for the overlay, then adds overlay
    '''

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
    ''' Writes down parameter data in a JSON file for later extraction after program reset'''
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
    ''' Reads parameter data in a JSON file for new program reset window'''
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

stream = cv2.VideoCapture(args.stream)  # Stream info from argparse

current_path     = os.getcwd()  # Get current working directory
overlay_path     = current_path + "/Alpha"  # Path to overlays
overlay_path_list= []  # List of images w\ full path
overlay_name_list= []  # List of images' names
valid_extensions = [ ".png" ]  # Allowable image extensions

for file in os.listdir( overlay_path ):  # Loop over images in directory
    extension    = os.path.splitext( file )[1]   # Get file's extensions

    if( extension.lower() not in valid_extensions ):  # If extensions is not in valid
        continue  # List, skip it.

    else:  # Else, append full file path
        overlay_path_list.append( os.path.join(overlay_path, file) )  # To list.
        overlay_name_list.append( os.path.splitext(file)[0] )                   

# Prepare overlay
global overlayImg, counter
counter     = 0  # Overlay switcher counter

if (args.overlay):  # If a overlay path is given
    overlayImg  = prepare_overlay( args["overlay"] )  # Prepare and process overlay
else:  # Else use defualt value in "prepare_overlay()" function
    overlayImg  = prepare_overlay()  # Prepare and process overlay

# If argparse setvalues flag is set trigger main loop
if(args.setvalues):
    setup_windows()  # Setup trackbar windows for main loop
    while(stream.isOpened()):
        ret, frame = stream.read()
        if ret == True:
            # Zooming into stream
            height,width,depth = frame.shape
            x, y, d = Center_update()
            frame = frame[y-d:y+d, x-d:x+d]
            frame = cv2.resize( frame, (640,480), interpolation=cv2.INTER_AREA )
            image = frame  # Copy for stacking
            (h, w) = frame.shape[:2]  # Determine width and height
            frame = np.dstack([ frame, np.ones((h, w),             # Stack the arrays in sequence, 
                                               dtype="uint8") ])   # depth-wise ( along the z-axis )
            overlay = np.zeros( ( h, w, 4 ), "uint8" )  # Empty np array w\ same dimensions as the frame
            proc, minHSV, maxHSV = procFrame( image, None, None )  # Procceses frame for detection and returns values
            params, detector = update_detector()  # Updates detector values from trackbar
            image, img_detected, parameters = find_pupil( proc, overlay, overlayImg, frame )  # Returns positional data and parameters

            cv2.imshow( 'ScaleBar', image )  # Show final output   
            cv2.imshow( 'BlobParamsBar', img_detected )  # Shows Blob detector window
            cv2.imshow( 'HSVbars', proc)  # Shows Rangefinder window
            if cv2.waitKey(1) & 0xFF == ord('r'):  # Press keystroke "r" to reset stream
                write_parameters( minHSV, maxHSV, parameters )
                print("Range Values")
                print(minHSV, maxHSV)
                print("Resetting program now")
                break
            elif cv2.waitKey(1) & 0xFF == ord('q'):  # Press keystroke "q" to quit stream
                print("Quitting now")
                break
        else:  # If video stream ends print values and exit
            print("Range Values")
            print(minHSV, maxHSV)
            print("Stream over, resetting program now")
            write_parameters( minHSV, maxHSV, parameters )
            break
        
    # Reset stream to reduce lag   
    stream.release()
    cv2.destroyAllWindows()           
    stream = cv2.VideoCapture(args.stream)

    setup_windows2()  # Setup final output window
    while(stream.isOpened()):
        ret, frame = stream.read()
        if ret == True:
            # Zooming into stream
            height,width,depth = frame.shape
            x, y, d = Center_update()
            frame = frame[y-d:y+d, x-d:x+d]
            frame = cv2.resize( frame, (640,480), interpolation=cv2.INTER_AREA )
            image = frame
            (h, w) = frame.shape[:2]  # Determine width and height
            frame = np.dstack([ frame, np.ones((h, w),                # Stack the arrays in sequence, 
                                               dtype="uint8")*255 ])  # depth-wise ( along the z-axis )
            overlay = np.zeros( ( h, w, 4 ), "uint8" )  # Empty np array w\ same dimensions as the frame
            parameters, minHSV, maxHSV = read_parameters()  # Reads parameters from Json
            proc, minHSV, maxHSV = procFrame( image, minHSV, maxHSV )  # Procceses frame for detection and returns values
            image, img_detected, parameters = find_pupil( proc, overlay, overlayImg, frame, parameters )  # Returns positional data and parameters

            cv2.imshow( 'ScaleBar', image )  # Show final output
            if cv2.waitKey(1) & 0xFF == ord('q'):  # Press "q" to quit stream
                print("Closing program now")
                stream.release()
                cv2.destroyAllWindows()
        else:  # Break loop if stream is over
            print("Stream over, closing program now")
            break

# Else read parameter data from Json file fefault and show only final window
else:
    setup_windows2()
    while(stream.isOpened()):
        ret, frame = stream.read()
        if ret == True:
            # Zooming into stream
            height,width,depth = frame.shape
            x, y, d = Center_update()
            frame = frame[y-d:y+d, x-d:x+d]
            frame = cv2.resize( frame, (640,480), interpolation=cv2.INTER_AREA )
            image = frame
            (h, w) = frame.shape[:2]  # Determine width and height
            frame = np.dstack([ frame, np.ones((h, w),                # Stack the arrays in sequence, 
                                               dtype="uint8")*255 ])  # depth-wise ( along the z-axis )
            overlay = np.zeros( ( h, w, 4 ), "uint8" )  # Empty np array w\ same dimensions as the frame
            parameters, minHSV, maxHSV = read_parameters()  # Reads parameters from Json
            proc, minHSV, maxHSV = procFrame( image, minHSV, maxHSV )  # Procceses frame for detection and returns values
            image, img_detected, parameters = find_pupil( proc, overlay, overlayImg, frame, parameters )  # Returns positional data and parameters
  
            cv2.imshow( 'ScaleBar', image )  # Show final output
            if cv2.waitKey(1) & 0xFF == ord('q'):  # Press "q" to quit stream
                print("Closing program now")
                break
        else:  # Break loop if stream is over
            print("Stream over, closing program now")
            break
            
stream.release()  # Releases stream after exit
cv2.destroyAllWindows()  # Destroys all windows after exit


          




