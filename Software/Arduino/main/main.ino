/*
 * Report back if device is at/beyond a threshold distance.
 * This sketch is used with the ESP32 that goes along with
 * the augmented ophthalmoscope/otoscope.
 * 
 * AUTHOR                   : Mohammad Odeh
 * WRITTEN                  : Aug. 6th, 2018 Year of Our Lord
 * LAST CONTRIBUTION DATE   : Aug. 6th, 2018 Year of Our Lord
 */

// Include required libraries
#include  <Wire.h>
#include  "Adafruit_VL6180X.h"

// Define various parameters
#define   BAUD          115200                            // Serial communications baudrate
#define   RETRIES       5                                 // Number of retries to initialize sensor
#define   THRESHOLD     15                                // Distance threshold we need to pass

// Define communication bytes
#define   SOH           0x01                              // Start of Header
#define   ENQ           0x05                              // Enquiry
#define   ACK           0x06                              // Acknowledged
#define   NAK           0x15                              // Not Acknowledged
#define   DC1           0x21                              // Device Control 1: Reboot system

Adafruit_VL6180X vl = Adafruit_VL6180X();                 // Instantiate sensor

#include  "get_dist.h"                                    // Get distance function
#include  "parseByte.h"                                   // Parse byte function

void setup() {
  bool    sensor_ready  = false  ;                        // Boolean to indicate that sensor was found
  uint8_t counter       = RETRIES;                        // Counter for number of attempts at initializing sensor
  
  Serial.begin( BAUD );                                   // Start serial
  while( !Serial );                                       // Wait until serial communication is established

  /* CASE: Sensor fails to initialize */
  if( !vl.begin() )                                       // Initialize sensor.
  {
    sensor_ready = false;                                 // Indicate that sensor is not ready
    while( !sensor_ready && counter!=0 )                  // Loop until device is initialized or limit is reached
    {
      sensor_ready = vl.begin();                          // vl.begin() returns 0 if sensor is ready
      delay( 1000 );                                      // Wait one second before attempting again
      counter--;                                          // Decrement counter
    }
    Serial.write( NAK ); delay( 250 );                    // Send an NAK to indicate system is NOT ready
    while( true )                                         // Get stuck in infinite loop and wait for fix
    {
      Serial.print( F("[INFO] SENSOR FAILURE\n") );       // [INFO] ...
      delay( 1000 );                                      // Delay 1sec so we don't spam serial
    }
  }

  /* CASE: Sensor succeeds to initialize */
  else
  { 
    Serial.write( SOH ); delay( 250 );                    // Send an SOH to indicate system is ready
    Serial.print( F("[INFO] SENSOR INITIALIZED\n") );     // [INFO] ...
  }
}

void loop() {
  get_dist();                                             // Perform readings
  if( Serial.available() ) parseByte();                   // Check on serial for incoming data
}
