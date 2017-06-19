'''
#
# A script to perform initial setup procedures for the TinyScreen version
# of the ophthalmoscope.
#
# AUTHOR : Mohammad Odeh
# DATE   : Jun. 19th, 2017
#
'''

import sys, os, platform, fileinput
from time import sleep
from os import path, listdir, lseek
from shutil import copy, copytree, rmtree, ignore_patterns

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
    print( ">>> System parameters already modified." )
    print( ">>> No further action is required.\n" )

# Else, copy config.txt to config.txt.BAK then modify
else:
    print( ">>> Copying config.txt ===to==> config.txt.BAK" )
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
                print( ">>> Replaced: %s" %TEXT_TO_SEARCH[i] )
                print( ">>> With    : %s\n" %TEXT_TO_INSERT[i] )

                # If index is out of bounds, do not increment counter
                if i == len(TEXT_TO_SEARCH)-1:
                    continue
                # Else increment counter
                else:
                    i=i+1

            # Else, keep original content
            else:
                output_file.write(line)
            
    print( ">>> Successfuly modified config.txt" )

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

# If autolaunch has been configured, do nothing
if (os.path.isfile(dst + "/launchOphto.sh")):
    print( ">>> Autolaunch on boot is already enabled." )
    print( ">>> No further action is required." )

# If autolaunch has NOT been configured, configure it
else:
    print( ">>> Configuring autolaunch." )
    
    # Copy launchOnBoot.py to the home directory
    copy(src, dst)

    # Append the launch command to the autostart file
    with open(PATH, "a") as f:
        f.write( "./launchOphto.sh\n") 
        f.close()

    print( ">>> Successfully appended to autostart." )
    print( ">>> Please reboot.\n" )

