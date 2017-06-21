'''
#
# A script to perform initial setup procedures for the TinyScreen version
# of the ophthalmoscope.
#
# AUTHOR    : Mohammad Odeh
# DATE      : Jun. 19th, 2017
# MODIFIED  : Jun. 20th, 2017
#
'''

import  sys, os, platform, argparse
from    os     import path, listdir, lseek
from    shutil import copy, copytree, rmtree, ignore_patterns

class TheErrorOfMoe(Exception):
    '''Raise this when something goes awry'''
    pass

# Construct Argument Parser
ap = argparse.ArgumentParser()
ap.add_argument("-u", "--update", action="store_true", required=False,
                help="Update current folder with all the files present in the Github repository")
args = vars(ap.parse_args())

# Enumerate strings to replace
TEXT_TO_SEARCH = [  "#overscan_left=16"         ,
                    "#overscan_right=16"        ,
                    "#overscan_top=16"          ,
                    "#overscan_bottom=16"       ,
                    "#framebuffer_width=1280"   ,
                    "#framebuffer_height=720"   ]

# Enumerate strings to insert
TEXT_TO_INSERT = [  "overscan_left=-10\n"       ,
                    "overscan_right=-10\n"      ,
                    "overscan_top=-24\n"        ,
                    "overscan_bottom=-24\n"     ,
                    "framebuffer_width=384\n"   ,
                    "framebuffer_height=288\n"  ]

print( '''
# ************************************************************************
#                       BACKUP & MODIFY config.txt FILE
# ************************************************************************
''' )

# Make sure we are defining the right directories
if platform.system() == "Linux":

    # Define useful paths
    file_path = "/boot"
    src = file_path + "/config.txt"
    dst = file_path + "/config.txt.BAK"

# If backup already exists, do nothing
if (os.path.isfile( dst )):
    print( ">>> Device already supports TFT." )
    print( ">>> No further action is required.\n" )

# Else, copy config.txt to config.txt.BAK then modify
else:
    print( ">>> Copying config.txt ====> config.txt.BAK" )
    copy(src, dst)
    print( ">>> Backup created." )
    print( ">>> Preparing to modify file.\n" )
    
    # Open file to read from, open file to write over
    with open(dst, 'r') as input_file, open(src, 'w') as output_file:
        i=0
        # Traverese file line-by-line
        for line in input_file:
            # If string is found, write over it
            if line.strip() == TEXT_TO_SEARCH[i]:
                output_file.write(TEXT_TO_INSERT[i])
                print( ">>> Found: %s" %TEXT_TO_SEARCH[i] )

                # If index is out of bounds, do not increment counter
                if i == len(TEXT_TO_SEARCH)-1:
                    continue
                # Else increment counter
                else:
                    i=i+1

            # Else, keep original content
            else:
                output_file.write(line)
            
    print( "\n>>> Successfuly modified config.txt" )
    print( ">>> Reboot required for changes to take effect.\n" )


print( '''
# ************************************************************************
#                               AUTOLAUNCH SCRIPT
# ************************************************************************
''')

# Make sure we are defining the right directories
if platform.system() == "Linux":

    # Define useful paths
    homeDir = "/home/pi"
    src = os.getcwd() + "/launchOphto.sh"
    dst = homeDir
    PATH = homeDir + "/.config/lxsession/LXDE-pi/autostart"


# Check if executable is in place
if (os.path.isfile(dst + "/launchOphto.sh")):
    print( ">>> Executable already in place.")
    isFile = True
else:
    # Copy launchOnBoot.py to the home directory
    isFile = False

# Check if string is appended to autolaunch
with open(PATH, 'r') as f:
    # Traverese file line-by-line
    for line in f:
        if line.strip() == "./launchOphto.sh":
            print( ">>> autostart entry found.")
            isString = True
            break
        else:
            isString = False


# If autolaunch has been configured, do nothing
if (isFile==True and isString==True):
    print( ">>> Autolaunch on boot is already enabled." )
    print( ">>> No further action is required." )


# Else if executable exists but not string, add string
elif (isFile==True and isString==False):
    print( ">>> Configuring autolaunch." )
    # Append the launch command to the autostart file
    with open(PATH, "a") as f:
        f.write( "./launchOphto.sh\n") 
        f.close()
        

# Else if string exists but executable doesn't, add executable
elif (isFile==False and isString==True):
    print( ">>> Configuring autolaunch." )
    copy(src, dst)


# Else if autolaunch has NOT been configured at all, configure it
elif (isFile==False and isString==False):
    print( ">>> Configuring autolaunch." )
    copy(src, dst)
    with open(PATH, "a") as f:
        f.write( "./launchOphto.sh\n") 
        f.close()

# Else, raise flag
else:
    raise TheErrorOfMoe("Schrodinger's autolaunch...")

if (isFile==False or isString==False):
    print( ">>> Successfully appended to autostart." )
    print( ">>> Please reboot.\n" )


print( '''
# ************************************************************************
#                       COPY FILES/DIRECTORIES + MODULES
# ************************************************************************
''' )

if platform.system()=='Linux':

    # Define useful paths
    homeDir = "/home/pi"
    src = os.getcwd()[:-9]
    dst = homeDir + "/Desktop/ophto"
    srcContent = listdir(src)

# If files have already been copied, do nothing
if (path.exists(dst)):
    print(">>> Ophthalmoscope Directory exists.\n>>> Checking files...")

    # If update argument is provided, update folder
    if args["update"]:
        print(">>> Cleaning up...")
        rmtree(dst)
        print(">>> Removed old files...")
        copytree(src, dst, ignore=ignore_patterns('*.pyc', 'tmp*'))
        print(">>> Updated current USER folder.")

    # Compare sizes of both folders
    dstContent = listdir(dst)
    if len(srcContent) > len(dstContent):          
        print(">>> The Github folder is ahead by %i file" %(len(srcContent)-len(dstContent)) )
        print(">>> Run this script using -u/--update to update USER folder.")
        print(">>> WARNING: Running the update WILL OVERWRITE current files.")
    else:
        print(">>> No further action is required.")
    

# If files have NOT been copied, copy them
else:
    print(">>> Copying files and directories...")
    
    # Copy files and directories to the home directory
    copytree(src, dst, ignore=ignore_patterns('*.pyc', 'tmp*'))

    print(">>> Files and directories have been successfuly copied.")

