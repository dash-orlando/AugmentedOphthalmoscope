'''

Crop image into a fixed aspect ratio of 1:1

Author: me
Date  : today

'''

# Import modules
from    time        import  sleep
import  numpy       as      np
import  cv2

# Read image
img = cv2.imread( "C:\\Users\\modeh\\Desktop\\eye_trainer\\haar_trainner\\data\\positive-clean\\image266.jpg" )

# Get image shape
global h, w
h, w = img.shape[:2]

# Get the position of the first non white pixel
# going from left to right ( x = minimum )
def find_x_min():
    print( "[INFO] Finding x=min(x)" )
    sleep( 0.5 )
    for x in range( 0, w ):        
        for y in range( 0, h ):
            pixel = img[y, x, 0]

            if( pixel < 250 ):
                x_min = x
                print( "[INFO] Found non white pixel at (x, y) = ({}, {})\n".format(x, y) )
                sleep( 0.5 )
                return( x_min )
            else:
                continue
    
            
# Get the position of the first non white pixel
# going from right to left ( x = maximum )
def find_x_max():
    print( "[INFO] Finding x=max(x)" )
    sleep( 0.5 )
    for x in range( w-1, 0 , -1):      
        for y in range( 0, h ):
            pixel = img[y, x, 0]

            if( pixel < 250 ):
                x_max = x
                print( "[INFO] Found non white pixel at (x, y) = ({}, {})\n".format(x, y) )
                sleep( 0.5 )
                return( x_max )
            else:
                continue

# Get the position of the first non white pixel
# going from top to bottom ( y = minimum )
def find_y_min():
    print( "[INFO] Finding y=min(y)" )
    sleep( 0.5 )
    for y in range( 0, h ):     
        for x in range( 0, w ):
            pixel = img[y, x, 0]

            if( pixel < 250 ):
                y_min = y
                print( "[INFO] Found non white pixel at (x, y) = ({}, {})\n".format(x, y) )
                sleep( 0.5 )
                return( y_min )
            else:
                continue

# Get the position of the first non white pixel
# going from bottom to top ( y = maximum )
def find_y_max():
    print( "[INFO] Finding y=max(y)" )
    sleep( 0.5 )
    for y in range( h-1, 0, -1 ):
        for x in range( 0, w ):
            pixel = img[y, x, 0]

            if( pixel < 250 ):
                y_max = y
                print( "[INFO] Found non white pixel at (x, y) = ({}, {})\n".format(x, y) )
                sleep( 0.5 )
                return( y_max )
            else:
                continue

# Call functions and get values
x_min = find_x_min()
x_max = find_x_max()
y_min = find_y_min()
y_max = find_y_max()

# Compute the ratio and pad if needed
dx = x_max - x_min
dy = y_max - y_min

if( dy > dx ):
    while( dy != dx ):
        x_max = x_max + 1
        dx = x_max - x_min
else:
    while( dx != dy ):
        y_max = y_max + 1
        dy = y_max - y_min

print( "[INFO] Aspect ratio is set to {}:{}".format(dx, dy) )

with open('positives.txt', 'a') as f:
    f.write( "cropped 1 %i %i %i %i\n" %(x_min, y_min, dx, dy) )

# Crop and save image to file
cropped_img = img[y_min:y_max, x_min:x_max]
DIR = "C:\\Users\\modeh\\Desktop\\cropped.jpg"
cv2.imwrite( DIR, cropped_img )
