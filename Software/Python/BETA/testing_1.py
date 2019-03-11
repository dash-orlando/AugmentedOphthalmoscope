import requests                                             # Allows you to send organic, grass-fed HTTP requests
import numpy                                    as np       # Always behind the scenes
import cv2                                                  # It's always a good idea to give vision to computers

# ************************************************************************
# =====================> DEFINE Camera Class <===========================*
# ************************************************************************

class Camera:
    
    def __init__(self, ip = '192.168.1.88', user = 'admin', pwd = 'admin'):     # default values are camera defaults
        
        self.ip = str(ip)
        self.user = str(user)
        self.pwd = str(pwd)

    def PTZ(self, action, count = 1):                                           # sends an http request with url to complete action
        
        ip = self.ip
        user = self.user
        pwd = self.pwd                                                          # actions are ZoomAdd, ZoomSub, FocusAdd, and FocusSub
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
        while(1):
            
            ret, frame = vcap.read()
            cv2.imshow('camera stream', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):                               # press keystroke "q" to break stream
                    
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

# ************************************************************************
# =====================> DEFINE Range_Finder Function <==================*
# ************************************************************************

# open the camera
cap = cv2.VideoCapture('output.avi')
range_filter = "HSV"

def callback(value):                                                                 # blank function for trackbar functionality
    pass

def setup_trackbars(range_filter):
    cv2.namedWindow("Trackbars", 0)

    for i in ["MIN", "MAX"]:
        v = 0 if i == "MIN" else 255

        for j in range_filter:
            cv2.createTrackbar("%s_%s" % (j, i), "Trackbars", v, 255, callback)

def get_trackbar_values(range_filter):
    values = []

    for i in ["MIN", "MAX"]:
        for j in range_filter:
            v = cv2.getTrackbarPos("%s_%s" % (j, i), "Trackbars")
            values.append(v)

    return values

def Range_Finder():
    setup_trackbars(range_filter)
     
    while True:

        #read the image from the camera
        ret, frame = cap.read()
        if ret == True:
            #You will need this later
            frame = cv2.cvtColor(frame, 35)
            #converting to HSV
            hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
            # get info from track bar and appy to result
            v1_min, v2_min, v3_min, v1_max, v2_max, v3_max = get_trackbar_values(range_filter)
            
            thresh = cv2.inRange(hsv, (v1_min, v2_min, v3_min), (v1_max, v2_max, v3_max))

            result = cv2.bitwise_and(frame,frame,mask = thresh)

            cv2.imshow('result', result)
            cv2.imshow("Original", frame)
            cv2.imshow("Thresh", thresh)

            # Quit the program when Q is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Minimum values", get_trackbar_values(range_filter)[:3])
                print("Maximum values", get_trackbar_values(range_filter)[3:])
                break
        else:
            print("Minimum values", get_trackbar_values(range_filter)[:3])
            print("Maximum values", get_trackbar_values(range_filter)[3:])
            break

    # When everything done, release the capture
    print 'closing program' 
    cap.release()
    cv2.destroyAllWindows()

##### ------------------------------------------------------------------------
def pupil_detect():
        
    cap = cv2.VideoCapture('output.avi')
    

    ##minHSV = np.array([h - thresh, s - thresh, v - thresh])
    ##maxHSV = np.array([h + thresh, s + thresh, v + thresh])
    ##
    ##thresholded = cv2.inRange(hsv, minHSV, maxHSV)
    ##outimage = cv2.bitwise_and(frame, frame, mask = thresholded)


    while(cap.isOpened()):
        
        ret, frame = cap.read()
        if ret == True:
            # Blur image to remove noise

         
            # Switch image from BGR colorspace to HSV
            hsv = cv2.cvtColor( frame, cv2.COLOR_BGR2HSV)
    ##        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            
            
            # define range of pipil color in HSV
##            h = 110
##            s = 170
##            v = 50
##            thresh = 30
##            minHSV = np.array([h - thresh, s - thresh, v - thresh])
##            maxHSV = np.array([h + thresh, s + thresh, v + thresh])
            minHSV = np.array([75, 10, 0])
            maxHSV = np.array([138, 255, 37])
            
            # Sets pixels to white if in black range, else will be set to black
            mask = cv2.inRange(hsv, minHSV, maxHSV)
            mask = cv2.GaussianBlur(mask, (5, 5), 0)




            parameters = cv2.SimpleBlobDetector_Params()                                # Parameters
             
            # Filter by Area.
            parameters.filterByArea = True                                              # ...
            parameters.minArea = np.pi * 15**2                                          # ...

            # Filter by color
            parameters.filterByColor = True                                             # ...
            parameters.blobColor = 255                                                    # ...

            # Filter by Circularity
            parameters.filterByCircularity = True                                       # ...
            parameters.minCircularity = .2                                            # ...
             
            # Filter by Convexity
            parameters.filterByConvexity = False                                         # ...
##            parameters.minConvexity = 0.1                                              # ...
                     
            # Filter by Inertia
            parameters.filterByInertia = True                                           # ...
            parameters.minInertiaRatio = 0.2                                           # ...

            # Distance Between Blobs
            parameters.minDistBetweenBlobs = 2000                                       # ...

            
            detector = cv2.SimpleBlobDetector_create( parameters )                      # Create detector

            keypoints = detector.detect(mask)
            img_detected = cv2.drawKeypoints(mask, keypoints, np.array([]), (0,255,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

            cv2.imshow('tracked image', img_detected)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()

#--------------------------------------------------

Range_Finder()
##pupil_detect()

          




