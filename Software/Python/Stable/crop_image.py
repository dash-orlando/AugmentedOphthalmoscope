'''
* 
* Crop image into a fixed aspect ratio of 1:1
* 
* CHANGELOG:
*   - Multiprocessing >>> Multithreading FTW!!!
* 
* AUTHOR                    :   Mohammad Odeh
* DATE                      :   Mar. 14th, 2018 Year de Nuestro Dios
* LAST CONTRIBUTION DATE    :   Mar. 15th, 2018 Year of Our Lord
'''

# Import modules
import  cv2                                                 # For saving images to disk
import  os, os.path                                         # OS specific tasks
import  numpy                       as      np              # For image cropping and maniupulation
from    time                        import  sleep           # Dormir for stability
from    multiprocessing             import  Process         # Spawn multiple processes
from    multiprocessing             import  Queue           # Queues for communication

# ************************************************************************
# =====================> DEFINE NECESSARY FUNCTIONS <=====================
# ************************************************************************

def find_x_min( Q ):
    '''
    Get the position of the first non white pixel
    going from left to right ( x = minimum )
    '''
    
    for x in range( 0, w ):        
        for y in range( 0, h ):
            pixel = img[y, x, 0]

            if( pixel < 250 ):
                x_min = x
                Q.put( x_min )
                return 1
            else:
                continue
    
# ------------------------------------------------------------------------


def find_x_max( Q ):
    '''
    Get the position of the first non white pixel
    going from right to left ( x = maximum )
    '''
    
    for x in range( w-1, 0 , -1):      
        for y in range( 0, h ):
            pixel = img[y, x, 0]

            if( pixel < 250 ):
                x_max = x
                Q.put( x_max )
                return 1
            else:
                continue

# ------------------------------------------------------------------------

def find_y_min( Q ):
    '''
    Get the position of the first non white pixel
    going from top to bottom ( y = minimum )
    '''
    for y in range( 0, h ):     
        for x in range( 0, w ):
            pixel = img[y, x, 0]

            if( pixel < 250 ):
                y_min = y
                Q.put( y_min )
                return 1
            else:
                continue

# ------------------------------------------------------------------------

def find_y_max( Q ):
    '''
    Get the position of the first non white pixel
    going from bottom to top ( y = maximum )
    '''
    
    for y in range( h-1, 0, -1 ):
        for x in range( 0, w ):
            pixel = img[y, x, 0]

            if( pixel < 250 ):
                y_max = y
                Q.put( y_max )
                return 1
            else:
                continue

# ************************************************************************
# ============================> SETUP PROGRAM <===========================
# ************************************************************************

# Create queues
Qx_min = Queue( maxsize=0 )                                                 # ...
Qx_max = Queue( maxsize=0 )                                                 # Queues for threads
Qy_min = Queue( maxsize=0 )                                                 # with no size limit
Qy_max = Queue( maxsize=0 )                                                 # ...

# Search directory for iamges
imageInDir  = "/home/pi/Desktop/haar_trainner/data/positive-clean"          # Specify directory where images are to be read from
imageOutDir = "/home/pi/Desktop/haar_trainner/data/positive-clean-cropped"  # Specify directory where images are to be stored to
fileName    = imageOutDir + "/positives.txt"                                # Annotated file name
image_path_list = []                                                        # Initialize empty list (for paths)
image_name_list = []                                                        # Initialize empty list (for names)
valid_image_extensions = [".jpg", ".jpeg"]                                  # Extensions of images we are interested in
valid_image_extensions = [item.lower() for item in valid_image_extensions]  # Put all extensions in lower case (just in case...*badum tsss*)

for file in os.listdir( imageInDir ):                                       # Loop over ALL files in the directory
    extension = os.path.splitext( file )[1]                                 # Get extension of each file
    
    if extension.lower() not in valid_image_extensions:                     # Check if the file's extension is in the valid list
        continue                                                            # ...
    else:                                                                   # If it is ...
        image_path_list.append( os.path.join(imageInDir, file) )            # Append directory  to list
        image_name_list.append( os.path.splitext(file)[0] )                 # Append image name to list

N_images = len(image_path_list)                                             # Get number of images
print( "[INFO] Found {} images".format( N_images ) )                        # Inform how many images were found
       
# ************************************************************************
# =========================> MAKE IT ALL HAPPEN <=========================
# ************************************************************************

global h, w                                                                 # Make variables accessible by entire script
from time import time
startTime = time()                                                          # Script performance profiling
i=0                                                                         # Counter for image names

# Read images and do da ting!
for imagePath in image_path_list:                                           # Loop over all the found images
    
    print( "[INFO] Processing image {}/{}".format( i+1, N_images ) )        # Progress report
    
    img = cv2.imread( imagePath )                                           # Read image (one-by-one)
    h, w = img.shape[:2]                                                    # Get image dimensions

    # Start threads
    Px_min = Process( target=find_x_min, args=( Qx_min, ) )                 # Start find_x_min thread
    Px_min.start()                                                          # ...
    
    Px_max = Process( target=find_x_max, args=( Qx_max, ) )                 # Start find_x_max thread
    Px_max.start()                                                          # ...
    
    Py_min = Process( target=find_y_min, args=( Qy_min, ) )                 # Start find_y_min thread
    Py_min.start()                                                          # ...
    
    Py_max = Process( target=find_y_max, args=( Qy_max, ) )                 # Start find_y_max thread
    Py_max.start()                                                          # ...
    
    # Get values
    x_min = Qx_min.get()                                                    # ...
    x_max = Qx_max.get()                                                    # Pull out values
    y_min = Qy_min.get()                                                    # stored in queue
    y_max = Qy_max.get()                                                    # ...
    
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

##    print( "[INFO] Aspect ratio is set to {}:{}".format(dx, dy) )

    with open(fileName, 'a') as f:
        f.write( "%s 1 %i %i %i %i\n" %(image_name_list[i], x_min, y_min, dx, dy) )

    # Crop and save image to file
    cropped_img = img[y_min:y_max, x_min:x_max]
    DIR =  "%s/%s.jpg" %( imageOutDir, image_name_list[i] )
    cv2.imwrite( DIR, cropped_img )
    i=i+1

print( "[INFO] Time to finish: {} s".format(time() - startTime) )

