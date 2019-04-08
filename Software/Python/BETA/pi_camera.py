from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import argparse

#-----setting up argparse

parser = argparse.ArgumentParser()
parser.add_argument("-res", "--resolution", nargs='*', default=[1920,1080],
                    type=int, help="the native resolution of the viewing screen, enter width,\
                    then height seperated by space\
                    (default set to 1920*1080)")
group = parser.add_mutually_exclusive_group()
group.add_argument("-full", "--fullscreen", help="runs the stream in fullscreen bordeless mode",
                    action="store_true")
group.add_argument("-half", "--halfscreen", help="runs the stream with window taking up half of the screen",
                    action="store_true")
group.add_argument("-c", "--custom", help="runs the stream with window resolution input, needs -x  and -y input",
                   action="store_true")
parser.add_argument("-x", nargs='?', type=int, help="the width of the window")
parser.add_argument("-y", nargs='?', type=int, help="the height of the window")
args = parser.parse_args()

#-----setup

# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 30
rawCapture = PiRGBArray(camera, size=(640,480))

# allow the camera to warmup
time.sleep(0.1)

# sets the native screen resolution, default 1920*1080
native_screen_res = (args.resolution[0], args.resolution[1])

# creates window for video depending on flag
if args.fullscreen:
    cv2.namedWindow("Video Stream", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Video Stream", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN )
elif args.halfscreen:
    cv2.namedWindow("Video Stream", cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow("Video Stream", native_screen_res[0]/2, native_screen_res[1])
    cv2.moveWindow("Video Stream", 0, 0)
elif args.custom:
    cv2.namedWindow("Video Stream", cv2.WINDOW_AUTOSIZE)
    cv2.resizeWindow("Video Stream", args.x, args.y)
else:
    cv2.namedWindow("Video Stream", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Video Stream", 640, 480)
    
#-----opencv display stream
    
# captures frames from the camera
for frame in camera.capture_continuous(rawCapture, format='bgr', use_video_port=True):
    # grabs the frame then initializes
    image = frame.array
    if args.fullscreen:
        # scales image to higher resolution
        image = cv2.resize(image, native_screen_res, cv2.INTER_AREA)
    elif args.halfscreen:
        # scales the image to half the native width keeping aspect ratio
        scale = (native_screen_res[0]/2)/image.shape[0]
        image = cv2.resize(image, (native_screen_res[0]/2, image.shape[1]*scale), cv2.INTER_AREA)
    elif args.custom:
        # scales image to set resolution
        image = cv2.resize(image, (args.x, args.y ), cv2.INTER_AREA)
    # show the frame     
    cv2.imshow("Video Stream", image)
    key = cv2.waitKey(1) & 0xFF
    # clear the stream for next frame
    rawCapture.truncate(0)
    # press keystroke "q" to break loop and stream
    if key == ord("q"):
        break
    if key == ord("Q"):
        break

cv2.destroyAllWindows()
