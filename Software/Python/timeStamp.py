"""
timeStamp.py

The following script/module/library had been built to create time-sensitive strings meant to be appended to data, folder names, or file names

Fluvio L. Lobo Fenoglietto 07/05/2016
"""

import time
import os

def calendarStamp():
    timeStamp = time.strftime("%Y-%m-%d")
    return timeStamp

def timeStamp():
    timeStamp = time.strftime("%H-%M-%S")
    return timeStamp

def fullStamp():
    timeStamp = time.strftime("%Y-%m-%d-%H-%M-%S")
    return timeStamp
    
def stampedFolder():
    folderName = "/" + fullStamp()
    return folderName


"""
References
1 - Print Time on Python - http://stackoverflow.com/questions/311627/how-to-print-date-in-a-regular-format-in-python
"""
