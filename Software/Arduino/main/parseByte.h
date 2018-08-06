/*
 * Read & process incoming serial data
 */
 
#define   SHORT_WAIT    250                               // Short wait in-between serial prints

void parseByte( ) {
  byte inByte = Serial.read();
  switch ( inByte )
  {
    // Systems Check
    case ENQ:
      Serial.write( ACK ); delay( SHORT_WAIT );
      break;

    // System Reboot
    case DC1:
      Serial.write( ACK ); delay( SHORT_WAIT );
      ESP.restart();
      break;
      
    default:
      Serial.write( NAK ); delay( SHORT_WAIT );
      Serial.print( F("INVALID COMMAND\n") );
      break;
  }
}
