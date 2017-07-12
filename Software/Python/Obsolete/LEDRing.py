# NeoPixel library strandtest example
# Author: Tony DiCola (tony@tonydicola.com)
#
# Direct port of the Arduino NeoPixel library strandtest example.  Showcases
# various animations on a strip of NeoPixels.
import time
import argparse
from neopixel import *

# Construct argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--overlay", required=False,
                help="path to overlay image")
ap.add_argument("-a", "--alpha", type=float, default=0.85,
                help="set alpha level (smaller = more transparent).\ndefault=0.85")
ap.add_argument("-d", "--debug", type=int, default=0,
                help="set flag equal to one (1) to enable debugging")
ap.add_argument("-b", "--brightness", type=int, default=155,
                help="set brightness level")
args = vars(ap.parse_args())

# LED strip configuration:
LED_COUNT      = 30                     # Number of LED pixels.
LED_PIN        = 18                     # GPIO pin connected to the pixels (must support PWM!).
LED_FREQ_HZ    = 800000                 # LED signal frequency in hertz (usually 800khz)
LED_DMA        = 5                      # DMA channel to use for generating signal (try 5)
LED_BRIGHTNESS = args["brightness"]     # Set to 0 for darkest and 255 for brightest
LED_INVERT     = False                  # True to invert the signal (when using NPN transistor level shift)
LED_CHANNEL    = 0
#LED_STRIP      = ws.SK6812_STRIP_RGBW	
LED_STRIP      = ws.SK6812W_STRIP

# Create NeoPixel object with appropriate configuration.
strip = Adafruit_NeoPixel(LED_COUNT, LED_PIN, LED_FREQ_HZ, LED_DMA, LED_INVERT, LED_BRIGHTNESS, LED_CHANNEL, LED_STRIP)
# Intialize the library (must be called once before other functions).
strip.begin()

# Define functions which animate LEDs in various ways.
def colorWipe(strip, color, wait_ms=50):
	"""Wipe color across display a pixel at a time."""
	for i in range(strip.numPixels()):
		strip.setPixelColor(i, color)
		strip.show()
		time.sleep(wait_ms/1000.0)

'''
# Main program logic follows:
if __name__ == '__main__':
	print ('Press Ctrl-C to quit.')
	while True:
		# Color wipe animations.
		colorWipe(strip, Color(255, 0, 0), 0)  # Red wipe
		time.sleep(2)
		colorWipe(strip, Color(0, 255, 0), 0)  # Blue wipe
		time.sleep(2)
		colorWipe(strip, Color(0, 0, 255), 0)  # Green wipe
		time.sleep(2)
		colorWipe(strip, Color(0, 0, 0, 255), 0)  # White wipe
		time.sleep(2)
		colorWipe(strip, Color(255, 255, 255), 0)  # Composite White wipe
		time.sleep(2)
		colorWipe(strip, Color(255, 255, 255, 255), 0)  # Composite White + White LED wipe
		time.sleep(2)
'''
