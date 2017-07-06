"""
usbProtocol.py

The following module has been created to manage the bluetooth interface between the control system and the connected devices

Michael Xynidis
Fluvio L Lobo Fenoglietto
09/01/2016


"""

# Import Libraries and/or Modules
#import bluetooth
"""
        Implementation of the "bluetooth" module may require the installation of the python-bluez package
        >> sudp apt-get install python-bluez
"""
import os
import serial
import time
from timeStamp import *

# Find RF Device
#   This function uses the hardware of the peripheral device or control system to scan/find bluetooth enabled devices
#   This function does not differenciate among found devices
#   Input   ::  None
#   Output  ::  {array, list} "availableDeviceNames", "availableDeviceBTAddresses"
def findDevices():
    print fullStamp() + " findDevices()"
    devices = bluetooth.discover_devices(
        duration=20,                                                                        # Search timeout
        lookup_names=True)                                                                  # Search and acquire names of antennas
    Ndevices = len(devices)                                                                 # Number of detected devices
    availableDeviceNames = []                                                               # Initialized arrays/lists for device names...
    availableDeviceBTAddresses = []                                                         # ...and their bluetooth addresses
    for i in range(0,Ndevices):                                                             # Populate device name and bluetooth address arrays/lists with a for-loop
        availableDeviceNames.append(devices[i][1])
        availableDeviceBTAddresses.append(devices[i][0])
    print fullStamp() + " Devices found (names): " + str(availableDeviceNames)              # Print the list of devices found
    print fullStamp() + " Devices found (addresses): " + str(availableDeviceBTAddresses)    # Print the list of addresses for the devices found
    return availableDeviceNames, availableDeviceBTAddresses                                 # Return arrays/lists of devices and bluetooth addresses

# Identify Smart Devices - General
#   This function searches through the list of detected devices and finds the smart devices corresponding to the input identifier
#   Input   ::  {string}     "smartDeviceIdentifier"
#           ::  {array/list} "availableDeviceNames", "availableDeviceBTAddresses"
#   Output  ::  {array/list} "smartDeviceNames", "smartDeviceBTAddresses"
def findSmartDevices(smartDeviceIdentifier, availableDeviceNames, availableDeviceBTAddresses):
    print fullStamp() + " findSmartDevices()"
    Ndevices = len(availableDeviceNames)
    smartDeviceNames = []
    smartDeviceBTAddresses = []
    for i in range(0,Ndevices):
        deviceIdentifier = availableDeviceNames[i][0:4]
        if deviceIdentifier == smartDeviceIdentifier:
            smartDeviceNames.append(availableDeviceNames[i])
            smartDeviceBTAddresses.append(availableDeviceBTAddresses[i])
    print fullStamp() + " Smart Devices found (names): " + str(smartDeviceNames)
    print fullStamp() + " Smart Devices found (addresses): " + str(smartDeviceBTAddresses)
    return smartDeviceNames, smartDeviceBTAddresses

# Identify Smart Device - Specific
#   This function searches through the list of detected devices and finds the specific smart device corresponding to the input name
#   Input   ::  {string}     "smartDeviceName"
#           ::  {array/list} "availableDeviceNames", "availableDeviceBTAddresses"
#   Output  ::  {array/list} "smartDeviceNames", "smartDeviceBTAddresses"
def findSmartDevice(smartDeviceName, availableDeviceNames, availableDeviceBTAddresses):
    print fullStamp() + " findSmartDevices()"
    Ndevices = len(availableDeviceNames)
    smartDeviceNames = []
    smartDeviceBTAddresses = []
    for i in range(0,Ndevices):
        deviceName = availableDeviceNames[i]
        if deviceName == smartDeviceName:
            smartDeviceNames.append(availableDeviceNames[i])
            smartDeviceBTAddresses.append(availableDeviceBTAddresses[i])
    print fullStamp() + " Smart Devices found (names): " + str(smartDeviceNames)
    print fullStamp() + " Smart Devices found (addresses): " + str(smartDeviceBTAddresses)
    return smartDeviceNames, smartDeviceBTAddresses                                                                      # Return RFObject or list of objects

# Create USB Port
def createPort(portNumber,baudrate,timeout):
    rfObject = serial.Serial(
        port = "/dev/ttyUSB" + str(portNumber),
        baudrate = baudrate,
        timeout = timeout)
    return rfObject

# Create USB Port
def createUSBPort(deviceName,portNumber,baudrate,attempts):
    print fullStamp() + " createUSBPort()"
    usbObject = serial.Serial(
        port = "/dev/ttyUSB" + str(portNumber),
        baudrate = baudrate)
    time.sleep(1)
    #usbConnectionCheck(usbObject,deviceName,portNumber,baudrate,attempts)
    usbObject.close()
    return usbObject

# Connection Check -Simple
#   Simplest variant of the connection check functions
def usbConnectionCheck(usbObject,deviceName,portNumber,baudrate,attempts):
    print fullStamp() + " usbConnectionCheck()"
    time.sleep(1)
    if usbObject.is_open == False:
        usbObject.open()
    print fullStamp() + " Requesting Device Name"
    usbObject.write('n')
    time.sleep(1)
    inString = usbObject.readline()[:-1]
    print inString
    if inString == deviceName:
        print fullStamp() + " Connection successfully established with " + deviceName
    else:
        usbObject.close()
        if attempts is not 0:
            return createUSBPort(deviceName,portNumber,baudrate,attempts-1)
        elif attempts is 0:
            print fullStamp() + " Connection Attempts Limit Reached"
            print fullStamp() + " Please troubleshoot " + deviceName

# Send Until ReaD
#       This function sends an input command through the rfcomm port to the remote device
#       The function sends such command persistently until a timeout or iteration check are met
#       Input   ::      rfObject                {object}        serial object
#                       outByte                 {chr}           command in characters/bytes
#                       timeout                 {int}           maximum wait time for serial communication
#                       iterCheck               {int}           maximum number of iterations for serial communication
#       Output  ::      inByte                  {chr}           response from remote device in characters/bytes
#                       terminal messages       {string}        terminal messages for logging      
def sendUntilRead(rfObject, outByte, timeout, iterCheck):
    print fullStamp() + " sendUntilRead()"                                                                  # Printing program name
    iterCount = 0
    startTime = time.time()                                                                                 # Initial time, instance before entering "while loop"
    while (time.time() - startTime) < timeout and iterCount <= iterCheck:                                   # While loop - will continue until either timeout or iteration check is reached
        print fullStamp() + " Communication attempt " + str(iterCount) + "/" + str(iterCheck)
        print fullStamp() + " Time = " + str(time.time()-startTime)
        rfObject.write(outByte)                                                                             # Send CHK / System Check request
        inByte = rfObject.read()                                                                            # Read response from remote device
        if inByte == definitions.ACK:                                                                       # If response equals ACK / Positive Acknowledgement
            # print fullStamp() + " ACK"                                                                    # Print terminal message, device READY / System Check Successful                                                                             
            return inByte                                                                                   # Return the byte read from the port
            break                                                                                           # Break out of the "while loop"
        elif inByte == definitions.NAK:                                                                     # If response equals NAK / Negative Acknowledgement
            # print fullStamp() + " NAK"                                                                    # Print terminal message, device NOT READY / System Check Failed
            return inByte                                                                                   # Return the byte read from the port
            break                                                                                           # Break out of the "while loop"
