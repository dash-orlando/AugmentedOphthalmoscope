import numpy as np
import cv2

stream = cv2.VideoCapture('fake_eye.avi', cv2.WINDOW_NORMAL
                          )

while(True):
    ret, frame = stream.read()
    cv2.imshow('frame', frame )
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

stream.release()
cv2.destroyAllWindows()
            
