"""
pyCam.py

The following script has been desveloped to control de Raspberry Pi camera module using Python

Fluvio L. Lobo Fenoglietto 06/23/2016
"""

# =======================
# Import Libraries and/or Modules
# =======================

import picamera

# =======================
# Variables
# =======================

picam = picamera.PiCamera()


# =======================
# Operation
# =======================
picam.start_preview()


"""
References
1- Raspberry Pi Camera Module Documentation - https://www.raspberrypi.org/documentation/usage/camera/python/README.md
"""
