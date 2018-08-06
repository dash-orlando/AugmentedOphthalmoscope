/*
 * Report back if device is at/beyond a threshold distance.
 */

uint8_t crnt_reading = 0;                                 // Current readings flag
uint8_t prvs_reading = 0;                                 // Previous readings flag

void get_dist() {
  uint8_t read_status = vl.readRangeStatus();             // Get status of readings

  if ( read_status == VL6180X_ERROR_NONE )                // If no errors are reported back, proceed
  {
    uint8_t range = vl.readRange();                       // Read the distance
    Serial.print( F("Range: ") ); Serial.println( range );// Print readings

    if ( range <= THRESHOLD )                             // Send true ONCE in case the state changes
    {
      crnt_reading = 1;                                   // Assign a value of true (1) to variable
      if ( crnt_reading != prvs_reading )                 // This makes sure that the status is only sent once
      {
        Serial.print( true );                             // Indicate that we are at/beyond threshold
        delay( 100 );                                     // Delay for stability
        prvs_reading = crnt_reading;                      // Update "previous values holder" with the current value
      }
    }

    else if (range > THRESHOLD)                           // Else, send false ONCE in case the state changes
    {
      crnt_reading = 0;                                   // Same ...
      if ( crnt_reading != prvs_reading )                 // as ...
      {
        Serial.print( false );                            // above ...
        delay( 100 );                                     // ...
        prvs_reading = crnt_reading;                      // ...
      }
    }

    // Else, dunno...
    else Serial.println( "ERROR" );
  }
}

